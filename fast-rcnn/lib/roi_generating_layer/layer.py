# --------------------------------------------------------
# Copyright (c) 2015 Stanford CVGL
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""The layer used during training to train a Fast R-CNN network.

RoIGeneratingLayer implements a Caffe Python layer.
"""

import math
import caffe
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps
import numpy as np
import yaml
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt

class RoIGeneratingLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""
    def _compute_targets_roi(self, rois, gts, gt_inds, fg_rois_per_this_image, labels):
        """Compute bounding-box regression targets for an image."""
        # Ensure ROIs are floats
        rois = rois.astype(np.float, copy=False)

        # Indices of examples for which we try to make predictions
        ex_inds = range(fg_rois_per_this_image)

        gt_rois = gts[gt_inds, 1:]
        ex_rois = rois[ex_inds, 1:]

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

    def _get_bbox_regression_labels_roi(self, bbox_target_data, num_classes):
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
        heatmap = bottom[0].data
        # (n, im, x1, y1, x2, y2) specifying an image batch index n, image index im and a rectangle (x1, y1, x2, y2)
        gts = bottom[1].data
        # class labels
        gt_labels = bottom[2].data
        # subclass labels
        gt_sublabels = bottom[3].data

        # image data
        im_blob = bottom[4].data

        # heatmap dimensions
        height = heatmap.shape[2]
        width = heatmap.shape[3]

        # generate all the boxes on the heatmap
        h = np.arange(height)
        w = np.arange(width)
        y, x = np.meshgrid(h, w, indexing='ij') 
        tmp = np.dstack((x, y))
        tmp = np.reshape(tmp, (-1, 2))
        num = tmp.shape[0]

        area = self._kernel_size * self._kernel_size
        aspect = [1, 0.75, 0.5, 0.25]  # height / width
        boxes = np.zeros((0, 4), dtype=np.float32)
        for i in xrange(len(aspect)):
            w = math.sqrt(area / aspect[i])
            h = w * aspect[i]
            x1 = np.reshape(tmp[:,0], (num,1)) - w * np.ones((num,1)) / 2
            x2 = np.reshape(tmp[:,0], (num,1)) + w * np.ones((num,1)) / 2
            y1 = np.reshape(tmp[:,1], (num,1)) - h * np.ones((num,1)) / 2
            y2 = np.reshape(tmp[:,1], (num,1)) + h * np.ones((num,1)) / 2
            boxes = np.vstack((boxes, np.hstack((x1, y1, x2, y2)) / self._spatial_scale))

        # compute box overlap with gt
        gt_boxes = gts[:,2:]
        #for i in xrange(gts.shape[0]):
        #    print '{:f} {:f} {:f} {:f} {:f} {:f}'.format(gts[i,0], gts[i,1], gts[i,2], gts[i,3], gts[i,4], gts[i,5])
        gt_overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))

        # number of ROIs
        image_ids = np.unique(gts[:,1])
        num_image = len(image_ids)
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_image
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        sublabels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * self._num_classes), dtype=np.float32)
        bbox_loss_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)

        # for each image
        for im in xrange(num_image):
            image_id = image_ids[im]

            # batches of this image
            index = np.where(gts[:,1] == image_id)[0]
            batch_ids = np.unique(gts[index,0])

            # number of objects in the image
            num_objs = index.size / batch_ids.size
            max_gt_overlaps = np.zeros((num_objs, 1), dtype=np.float32)
            print 'image {:d}, {:d} objects'.format(int(image_id), int(num_objs))

            # for each batch (one scale of an image)
            boxes_fg = np.zeros((0, 6), dtype=np.float32)
            boxes_bg = np.zeros((0, 6), dtype=np.float32)
            gt_inds_fg = np.zeros((0), dtype=np.int32)
            for i in xrange(len(batch_ids)):
                batch_id = batch_ids[i]

                # compute max overlap
                index_batch = np.where(gts[:,0] == batch_id)[0]
                overlaps = gt_overlaps[:,index_batch]
                max_overlaps = overlaps.max(axis = 1)
                argmax_overlaps = overlaps.argmax(axis = 1)

                """ debuging
                # show image
                im = im_blob[batch_id, :, :, :].transpose((1, 2, 0)).copy()
                im += cfg.PIXEL_MEANS
                im = im[:, :, (2, 1, 0)]
                im = im.astype(np.uint8)
                plt.imshow(im)

                # draw boxes
                for j in xrange(len(index_batch)):
                    roi = gt_boxes[index_batch[j],:]
                    plt.gca().add_patch(
                        plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                                       roi[3] - roi[1], fill=False,
                                       edgecolor='r', linewidth=3))

                inds = np.where(max_overlaps > 0.7)[0]
                for j in xrange(len(inds)):
                    roi = boxes[inds[j],:]
                    plt.gca().add_patch(
                        plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                                       roi[3] - roi[1], fill=False,
                                       edgecolor='g', linewidth=3))
                plt.show()
                """

                tmp = np.reshape(overlaps.max(axis = 0), (-1, 1))
                max_gt_overlaps = np.reshape(np.hstack((max_gt_overlaps, tmp)).max(axis = 1), (num_objs,1))
            
                # extract max scores
                scores = heatmap[batch_id]
                max_scores = np.reshape(scores[1:].max(axis = 0), (1,-1))
                max_scores = np.tile(max_scores, len(aspect)).transpose()

                # collect positives
                fg_inds = np.where(max_overlaps > cfg.TRAIN.FG_THRESH)[0]
                batch_ind = batch_id * np.ones((fg_inds.shape[0], 1))
                boxes_fg = np.vstack((boxes_fg, np.hstack((batch_ind, boxes[fg_inds,:], max_scores[fg_inds]))))
                gt_inds_fg = np.hstack((gt_inds_fg, index_batch[argmax_overlaps[fg_inds]]))

                # flags[argmax_overlaps[fg_inds]] = 1

                # collect negatives
                bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) & (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
                batch_ind = batch_id * np.ones((bg_inds.shape[0], 1))
                boxes_bg = np.vstack((boxes_bg, np.hstack((batch_ind, boxes[bg_inds,:], max_scores[bg_inds]))))

            print max_gt_overlaps

            # find hard positives
            # sort scores and indexes
            I = np.argsort(boxes_fg[:,5])
            # number of fg in the image
            fg_rois_per_this_image = int(np.minimum(fg_rois_per_image, I.size))
            I = I[0:fg_rois_per_this_image]
            boxes_fg = boxes_fg[I,:]
            gt_inds_fg = gt_inds_fg[I]

            # find hard negatives
            # sort scores and indexes descending
            I = np.argsort(boxes_bg[:,5])[::-1]
            # number of bg in the image
            bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
            bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, I.size)
            I = I[0:bg_rois_per_this_image]
            boxes_bg = boxes_bg[I,:]

            # ROIs
            rois = np.vstack((boxes_fg[:,:5], boxes_bg[:,:5]))

            # compute information of the ROIs: labels, sublabels, bbox_targets, bbox_loss_weights
            length = rois.shape[0]

            labels = np.zeros((length), dtype=np.float32)
            labels[0:fg_rois_per_this_image] = gt_labels[gt_inds_fg]

            sublabels = np.zeros((length), dtype=np.float32)
            sublabels[0:fg_rois_per_this_image] = gt_sublabels[gt_inds_fg]

            # first try without target normalization
            bbox_targets_data = self._compute_targets_roi(rois, gts, gt_inds_fg, fg_rois_per_this_image, labels)
            bbox_targets, bbox_loss = self._get_bbox_regression_labels_roi(bbox_targets_data, self._num_classes)

            # Add to RoIs blob
            rois_blob = np.vstack((rois_blob, rois))

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

