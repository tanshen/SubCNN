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
        
        # show image
        for i in xrange(rois_blob.shape[0]):
            batch_id = int(rois_blob[i,0])
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
        #""

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
