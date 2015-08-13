# --------------------------------------------------------
# Copyright (c) 2015 Stanford CVGL
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""The layer used during training to train a Fast R-CNN network.

RoIGeneratingLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps
import numpy as np
import yaml
from multiprocessing import Process, Queue

class RoIGeneratingLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def setup(self, bottom, top):
        """Setup the RoIGeneratingLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']
        self._kernel_size = layer_params['kernel_size']
        self._spatial_scale = layer_params['spatial_scale']

        self._name_to_top_map = {
            'rois': 0,
            'labels': 1}

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top[1].reshape(1)

        if cfg.TRAIN.BBOX_REG:
            self._name_to_top_map['bbox_targets'] = 2
            self._name_to_top_map['bbox_loss_weights'] = 3

            # bbox_targets blob: R bounding-box regression targets with 4
            # targets per class
            top[2].reshape(1, self._num_classes * 4)

            # bbox_loss_weights blob: At most 4 targets per roi are active;
            # thisbinary vector sepcifies the subset of active targets
            top[3].reshape(1, self._num_classes * 4)

        # add subclass labels
        if cfg.TRAIN.SUBCLS:
            self._name_to_top_map['sublabels'] = 4
            top[4].reshape(1)

    def forward(self, bottom, top):
        # parse input
        heatmap = bottom[0]
        # (n, x1, y1, x2, y2) specifying an image batch index n and a rectangle (x1, y1, x2, y2)
        gts = bottom[1]
        # class labels
        gt_labels = bottom[2]
        # subclass labels
        gt_sublabels = bottom[3]

        # heatmap dimensions
        num_image = heatmap.shape[0]
        height = heatmap.shape[2]
        width = heatmap.shape[3]

        # generate all the boxes on the heatmap
        h = np.arange(height)
        w = np.arange(width)
        y, x = np.meshgrid(h, w, indexing='ij') 
        tmp = np.dstack((x, y))
        tmp = np.reshape(tmp, (-1, 2))
        boxes = np.hstack((tmp - (self._kernel_size-1)*np.ones(tmp.shape)/2, tmp + (self._kernel_size-1)*np.ones(tmp.shape)/2)) / self._spatial_scale

        # compute box overlap with gt
        gt_boxes = gts[:,1:4]
        gt_overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))

        # number of ROIs
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_image
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        sublabels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * self._num_classes), dtype=np.float32)
        bbox_loss_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)

        # for each image
        for i in xrange(num_image):

            # compute max overlap
            index_gt = np.where(gts[:,0] == i)[0]
            overlaps = gt_overlaps[:,index_gt]
            max_overlaps = overlaps.max(axis = 1)
            argmax_overlaps = overlaps.argmax(axis = 1)
            
            # extract max scores
            max_scores = np.reshape(heatmap[i,1:].max(axis = 0), -1)

            # find hard positives
            fg_inds = np.where(max_overlaps > cfg.TRAIN.FG_THRESH)[0]
            # sort scores and indexes
            I = np.argsort(max_scores[fg_inds])
            fg_inds = fg_inds[I]
            # number of fg in the image
            fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
            fg_inds = fg_inds[0:fg_rois_per_this_image]

            # find hard negatives
            bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) & (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
            if bg_inds.size == 0:
                bg_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH_HI)[0]
            # sort scores and indexes descending
            I = np.argsort(max_scores[bg_inds])[::-1]
            bg_inds = bg_inds[I]
            # number of bg in the image
            bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
            bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_inds.size)
            bg_inds = bg_inds[0:bg_rois_per_this_image]

            # ROIs
            keep_inds = np.append(fg_inds, bg_inds)
            rois = boxes[keep_inds, :]

            # compute information of the ROIs: labels, sublabels, bbox_targets, bbox_loss_weights
            gt_inds = index_gt[argmax_overlaps[fg_inds]]

            labels = np.zeros(keep_inds.shape, dtype=np.float32)
            labels[0:fg_rois_per_this_image] = gt_labels[gt_inds]

            sublabels = np.zeros(keep_inds.shape, dtype=np.float32)
            sublabels[0:fg_rois_per_this_image] = gt_sublabels[gt_inds]

            # first try without target normalization
            bbox_targets_data = _compute_targets(rois, gts, gt_inds, fg_rois_per_this_image, labels)
            bbox_targets, bbox_loss = _get_bbox_regression_labels(bbox_targets_data, self._num_classes)

            # Add to RoIs blob
            batch_ind = i * np.ones((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))

            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            sublabels_blob = np.hstack((sublabels_blob, sublabels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_loss_blob = np.vstack((bbox_loss_blob, bbox_loss))

        # copy blobs into this layer's top blob vector
        blobs = {'rois': rois_blob,
                 'labels': labels_blob}
        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_loss_weights'] = bbox_loss_blob
        if cfg.TRAIN.SUBCLS:
            blobs['sublabels'] = sublabels_blob

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)
            

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _compute_targets(rois, gts, gt_inds, fg_rois_per_this_image, labels):
        """Compute bounding-box regression targets for an image."""
        # Ensure ROIs are floats
        rois = rois.astype(np.float, copy=False)

        # Indices of examples for which we try to make predictions
        ex_inds = range(fg_rois_per_this_image)

        gt_rois = gts[gt_inds, 1:]
        ex_rois = rois[ex_inds, :]

        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + cfg.EPS
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + cfg.EPS
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + cfg.EPS
        gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + cfg.EPS
        gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = np.log(gt_widths / ex_widths)
        targets_dh = np.log(gt_heights / ex_heights)

        targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
        targets[ex_inds, 0] = labels[ex_inds]
        targets[ex_inds, 1] = targets_dx
        targets[ex_inds, 2] = targets_dy
        targets[ex_inds, 3] = targets_dw
        targets[ex_inds, 4] = targets_dh
        return targets

    def _get_bbox_regression_labels(bbox_target_data, num_classes):
        """Bounding-box regression targets are stored in a compact form in the roidb.

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets). The loss weights
        are similarly expanded.

        Returns:
            bbox_target_data (ndarray): N x 4K blob of regression targets
            bbox_loss_weights (ndarray): N x 4K blob of loss weights
        """
        clss = bbox_target_data[:, 0]
        bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
        bbox_loss_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
        inds = np.where(clss > 0)[0]
        for ind in inds:
            cls = clss[ind]
            start = 4 * cls
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_loss_weights[ind, start:end] = [1., 1., 1., 1.]
        return bbox_targets, bbox_loss_weights
