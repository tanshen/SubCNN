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
# import matplotlib.pyplot as plt

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
            'rois': 0}

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)


    def forward(self, bottom, top):
        # parse input
        heatmap = bottom[0].data
        # boxes on the grid
        boxes = bottom[1].data

        # build the region of interest
        rois_blob = np.zeros((0, 5), dtype=np.float32)

        # for each scale of the image
        for im in xrange(heatmap.shape[0]):

            scores = heatmap[im]
            max_scores = np.reshape(scores[1:].max(axis = 0), (1,-1))
            max_scores = np.tile(max_scores, len(cfg.TRAIN.ASPECTS)).transpose()

            # collect boxes with score larger than threshold
            fg_inds = np.where(max_scores > cfg.TEST.ROI_THRESHOLD)[0]
            batch_ind = im * np.ones((fg_inds.shape[0], 1))
            rois_blob = np.vstack((rois_blob, np.hstack((batch_ind, boxes[fg_inds,:]))))

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

        # copy blobs into this layer's top blob vector
        blobs = {'rois': rois_blob}

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

