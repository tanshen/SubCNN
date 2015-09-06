# --------------------------------------------------------
# Copyright (c) 2015 Stanford CVGL
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""The layer used during testing of a Fast R-CNN network.

RoITestingLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps
import numpy as np
import yaml
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt

class RoITestingLayer(caffe.Layer):
    """Fast R-CNN layer used for testing."""

    def setup(self, bottom, top):
        """Setup the RoITestingLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']
        self._kernel_size = layer_params['kernel_size']
        self._spatial_scale = layer_params['spatial_scale']

        self._name_to_top_map = {
            'rois': 0,
            'rois_sub': 1}

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)
        top[1].reshape(1, 5)


    def forward(self, bottom, top):
        # parse input
        heatmap = bottom[0].data
        # boxes on the grid
        boxes = bottom[1].data

        # build the region of interest
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        rois_sub_blob = np.zeros((0, 5), dtype=np.float32)
        roi_max = np.zeros((1, 5), dtype=np.float32)
        roi_max_map = np.zeros((1, 5), dtype=np.float32)
        roi_score = 0

        # for each scale of the image
        for i in xrange(heatmap.shape[0]):

            max_scores = np.reshape(heatmap[i], (1,-1))
            max_scores = np.repeat(max_scores, len(cfg.TRAIN.ASPECTS)).transpose()
            assert (max_scores.shape[0] == boxes.shape[0])

            # collect boxes with score larger than threshold
            fg_inds = np.where(max_scores > cfg.TEST.ROI_THRESHOLD)[0]
            batch_ind = i * np.ones((fg_inds.shape[0], 1))
            rois_sub_blob = np.vstack((rois_sub_blob, np.hstack((batch_ind, boxes[fg_inds,:]))))

            # scale index of this batch is i
            scale_ind = i
            scale = cfg.TEST.SCALES[scale_ind]
            scale_ind_map = cfg.TEST.SCALE_MAPPING[scale_ind]
            scale_map = cfg.TEST.SCALES[scale_ind_map]
            batch_ind_map = scale_ind_map * np.ones((fg_inds.shape[0], 1))
            rois_blob = np.vstack((rois_blob, np.hstack((batch_ind_map, boxes[fg_inds,:] * scale_map/scale))))

            if np.max(max_scores) > roi_score:
                roi_score = np.max(max_scores)
                ind = np.argmax(max_scores)
                roi_max[0,0] = i
                roi_max[0,1:] = boxes[ind,:]
                roi_max_map[0,0] = scale_ind_map
                roi_max_map[0,1:] = boxes[ind,:] * scale_map/scale

            """ debuging
            print boxes.shape, heatmap.shape, max(max_scores), fg_inds.shape, fg_inds
            # show image
            im_blob = bottom[2].data
            im = im_blob[i, :, :, :].transpose((1, 2, 0)).copy()
            im += cfg.PIXEL_MEANS
            im = im[:, :, (2, 1, 0)]
            im = im.astype(np.uint8)
            plt.imshow(im)

            # draw boxes
            for j in xrange(len(fg_inds)):
                roi = boxes[fg_inds[j],:]
                plt.gca().add_patch(
                    plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                                   roi[3] - roi[1], fill=False,
                                   edgecolor='g', linewidth=3))
            plt.show()
            #"""

        # prevent empty roi
        if rois_sub_blob.shape[0] == 0:
            rois_sub_blob = roi_max
            rois_blob = roi_max_map

        # copy blobs into this layer's top blob vector
        blobs = {'rois': rois_blob,
                 'rois_sub': rois_sub_blob}
        print rois_blob.shape

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

