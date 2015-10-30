# --------------------------------------------------------
# Copyright (c) 2015 Stanford CVGL
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""The layer used during training to train a Fast R-CNN network.

RoIVisualizingLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
import numpy as np
import matplotlib.pyplot as plt

class RoIVisualizingLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def setup(self, bottom, top):
        pass


    def forward(self, bottom, top):
        # parse input
        rois_blob = bottom[0].data
        im_blob = bottom[1].data
        sublabels = bottom[2].data

        num_scale_base = len(cfg.TRAIN.SCALES_BASE)
        num_scale = len(cfg.TRAIN.SCALES)
        
        # show image
        for i in xrange(rois_blob.shape[0]):
            if sublabels[i] == 0:
                break

            batch_id = int(rois_blob[i,0])

            index_image = batch_id / num_scale
            index_scale = batch_id % num_scale
            scale = cfg.TRAIN.SCALES[index_scale]
            index_scale_base = index_scale / cfg.TRAIN.NUM_PER_OCTAVE
            scale_base = cfg.TRAIN.SCALES_BASE[index_scale_base]
            batch_index = index_image * num_scale_base + index_scale_base

            im = im_blob[batch_index, :, :, :].transpose((1, 2, 0)).copy()
            im += cfg.PIXEL_MEANS
            im = im[:, :, (2, 1, 0)]
            im = im.astype(np.uint8)
            plt.imshow(im)

            # draw boxes
            roi = rois_blob[i,1:] * scale_base / scale
            print roi
            plt.gca().add_patch(
                plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                               roi[3] - roi[1], fill=False,
                               edgecolor='g', linewidth=3))
            plt.show()
        #""

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
