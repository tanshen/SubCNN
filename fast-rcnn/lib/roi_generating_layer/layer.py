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
import random

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
            'rois_sub': 1,
            'labels': 2}

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)
        top[1].reshape(1, 5)

        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top[2].reshape(1)

        if cfg.TRAIN.BBOX_REG:
            self._name_to_top_map['bbox_targets'] = 3
            self._name_to_top_map['bbox_loss_weights'] = 4

            # bbox_targets blob: R bounding-box regression targets with 4
            # targets per class
            top[3].reshape(1, self._num_classes * 4)

            # bbox_loss_weights blob: At most 4 targets per roi are active;
            # thisbinary vector sepcifies the subset of active targets
            top[4].reshape(1, self._num_classes * 4)

        # add subclass labels
        if cfg.TRAIN.SUBCLS:
            self._name_to_top_map['sublabels'] = 5
            top[5].reshape(1)

    def forward(self, bottom, top):
        # parse input
        conv_sub_prob = bottom[0].data
        info_boxes = bottom[1].data

        # compute the heatmap
        heatmap = conv_sub_prob[:, 1:, :, :].max(axis = 1)

        # numbers
        num_batch = heatmap.shape[0]
        num_scale = len(cfg.TRAIN.SCALES)
        num_aspect = len(cfg.TRAIN.ASPECTS)
        num_image = num_batch / num_scale
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_image
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

        # process the positive boxes
        num_positive = info_boxes.shape[0]
        scores_positive = np.zeros((num_positive), dtype=np.float32)
        sep_positive = -1 * np.ones((num_image+1), dtype=np.int32)
        for i in xrange(num_positive):
            cx = int(info_boxes[i, 0])
            cy = int(info_boxes[i, 1])
            batch_index = int(info_boxes[i, 2])
            scores_positive[i] = heatmap[batch_index, cy, cx]
            # mask the heatmap location
            heatmap[batch_index, cy, cx] = -1.0
            # check which image
            image_index = int(batch_index / num_scale)
            sep_positive[image_index+1] = i

        # select positive boxes for each image
        index_positive = []
        count_image = np.zeros((num_image), dtype=np.int32)
        for i in xrange(num_image):
            num = sep_positive[i+1] - sep_positive[i]
            index = np.array(range(sep_positive[i]+1, sep_positive[i+1]+1))
            if num <= fg_rois_per_image:
                # use all the positives of this image
                index_positive.extend(index)
                count_image[i] = len(index)
            else:
                # select hard positives (low score positives)
                scores = scores_positive[index]
                I = np.argsort(scores)
                index_positive.extend(index[I[0:fg_rois_per_image]])
                count_image[i] = fg_rois_per_image

        # select negative boxes for each image
        index_negative = []
        for i in xrange(num_image):
            batch_index = range(i * num_scale, (i+1) * num_scale)
            # sort heatmap to select hard negatives (high score negatives)
            I = np.argsort(heatmap[batch_index], axis=None)[::-1]
            num = rois_per_image - count_image[i]
            index_negative.extend(I[0:num] + i * num_scale * heatmap.shape[1] * heatmap.shape[2])

        # build the blobs of interest
        batch_size = cfg.TRAIN.BATCH_SIZE
        rois_blob = np.zeros((batch_size, 5), dtype=np.float32)
        rois_sub_blob = np.zeros((batch_size, 5), dtype=np.float32)
        labels_blob = np.zeros((batch_size), dtype=np.float32)
        sublabels_blob = np.zeros((batch_size), dtype=np.float32)
        bbox_targets_blob = np.zeros((batch_size, 4 * self._num_classes), dtype=np.float32)
        bbox_loss_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)

        count = 0
        # positives
        for i in xrange(len(index_positive)):
            ind = index_positive[i]
            rois_sub_blob[count,:] = info_boxes[ind, 2:7]
            rois_blob[count,:] = info_boxes[ind, 7:12]
            labels_blob[count] = info_boxes[ind, 12]
            sublabels_blob[count] = info_boxes[ind, 13]
            # bounding box regression
            cls = int(info_boxes[ind, 12])
            start = 4 * cls
            end = start + 4
            bbox_targets_blob[count, start:end] = info_boxes[ind, 14:]
            bbox_loss_blob[count, start:end] = [1., 1., 1., 1.]
            count += 1

        # negatives
        for i in xrange(len(index_negative)):
            ind = index_negative[i]
            # parse index
            batch_index = int(ind / (heatmap.shape[1] * heatmap.shape[2]))
            tmp = ind % (heatmap.shape[1] * heatmap.shape[2])
            cy = int(tmp / heatmap.shape[2])
            cx = tmp % heatmap.shape[2]
            # sample an aspect ratio
            aspect_index = random.randint(1, num_aspect) - 1
            width = cfg.TRAIN.ASPECT_WIDTHS[aspect_index]
            height = cfg.TRAIN.ASPECT_HEIGHTS[aspect_index]
            # scale mapping
            image_index = int(batch_index / num_scale)
            scale_index = batch_index % num_scale
            scale = cfg.TRAIN.SCALES[scale_index]
            # check if the point is inside this scale
            rescale = scale / cfg.TRAIN.SCALES[-1]
            if cx < heatmap.shape[2] * rescale and cy < heatmap.shape[1] * rescale:
                scale_index_map = cfg.TRAIN.SCALE_MAPPING[scale_index]
                scale_map = cfg.TRAIN.SCALES[scale_index_map]
                batch_index_map = image_index * num_scale + scale_index_map
            else:
                # do not do scale mapping
                scale_map = scale
                batch_index_map = batch_index
            # assign information
            rois_sub_blob[count, 0] = batch_index
            rois_sub_blob[count, 1] = (cx - width / 2) / cfg.TRAIN.SPATIAL_SCALE
            rois_sub_blob[count, 2] = (cy - height / 2) / cfg.TRAIN.SPATIAL_SCALE
            rois_sub_blob[count, 3] = (cx + width / 2) / cfg.TRAIN.SPATIAL_SCALE
            rois_sub_blob[count, 4] = (cy + height / 2) / cfg.TRAIN.SPATIAL_SCALE
            rois_blob[count, 0] = batch_index_map
            rois_blob[count, 1:] = rois_sub_blob[count, 1:] * scale_map / scale
            count = count + 1

        assert (count == cfg.TRAIN.BATCH_SIZE)

        """ debuging
        # show image
        import matplotlib.pyplot as plt
        im_blob = bottom[2].data
        for i in xrange(rois_blob.shape[0]):
            batch_id = rois_blob[i,0]
            im = im_blob[batch_id, :, :, :].transpose((1, 2, 0)).copy()
            im += cfg.PIXEL_MEANS
            im = im[:, :, (2, 1, 0)]
            im = im.astype(np.uint8)
            plt.imshow(im)

            # draw boxes
            roi = rois_blob[i,1:]
            print roi
            plt.gca().add_patch(
                plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                               roi[3] - roi[1], fill=False,
                               edgecolor='g', linewidth=3))
            plt.show()
        #"""

        # copy blobs into this layer's top blob vector
        blobs = {'rois': rois_blob,
                 'rois_sub': rois_sub_blob,
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
        # Initialize all the gradients to 0. We will accumulate gradient
        bottom[0].diff[...] = np.zeros_like(bottom[0].data)

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

