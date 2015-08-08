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
import numpy as np
import yaml
from multiprocessing import Process, Queue

class RoIGeneratingLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._kernel_size = layer_params['kernel_size']
        self._spatial_scale = layer_params['spatial_scale']

    def forward(self, bottom, top):
        heatmap = bottom[0]
        height = heatmap.shape[2]
        width = heatmap.shape[3]

        # generate all the boxes on the heatmap
        h = np.arange(height)
        w = np.arange(width)
        y, x = np.meshgrid(h, w, indexing='ij') 
        tmp = np.dstack((y, x))
        tmp = np.reshape(tmp, (-1, 2))
        boxes = np.hstack((tmp, self._kernel_size * np.ones(tmp.shape)))

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
