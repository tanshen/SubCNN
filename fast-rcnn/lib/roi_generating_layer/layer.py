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
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._kernel_size = layer_params['kernel_size']
        self._spatial_scale = layer_params['spatial_scale']

    def forward(self, bottom, top):
        # parse input
        heatmap = bottom[0]
        # (n, x1, y1, x2, y2) specifying an image batch index n and a rectangle (x1, y1, x2, y2)
        gts = bottom[1]
        # subclass labels (filter index)
        gt_sublabels = bottom[2]

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

        # for each image
        for i in xrange(num_image):
            # extract ROIs
            keep_inds = []

            # compute max overlap
            index_gt = np.where(gts[:,0] == i)[0]
            overlaps = gt_overlaps[:,index_gt]
            max_overlaps = overlaps.max(axis = 1)
            arg_overlaps = overlaps.argmax(axis = 1)

            # find hard positives
            fg_inds = np.where(max_overlaps > cfg.TRAIN.FG_THRESH)[0]
            # extract scores of the positives
            sublabels = gt_sublabels[arg_overlaps[fg_inds]]
            

        # extract ROIs
        keep_inds = []
        # for each positive gt box
        for i in xrange(gts.shape[0]):
            index_image = gts[i,0]
            sublabel = gt_sublabels[i]
            # box scores
            scores = np.reshape(heatmap[index_image, sublabel, y, x], -1)
            # find hard positives
            overlaps = gt_overlaps[:,i]
            fg_inds = np.where(overlaps > cfg.TRAIN.FG_THRESH)[0]
            # sort scores and indexes
            I = np.argsort(scores[fg_inds])
            fg_inds = fg_inds[I]
            # number of fg in the image
            num_gt = len(np.where(gts[:,0] == index_image)[0])
            rois_per_gt = np.round(rois_per_image / num_gt)
            fg_rois_per_gt = np.round(fg_rois_per_image / num_gt)
            fg_rois_per_this_gt = np.minimum(fg_rois_per_gt, fg_inds.size)
            keep_inds.append(fg_inds[0:fg_rois_per_this_gt])

            # find hard negatives
            bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
            if bg_inds.size == 0:
                bg_inds = np.where(overlaps < cfg.TRAIN.BG_THRESH_HI)[0]
            # sort scores and indexes descending
            I = np.argsort(scores[bg_inds])[::-1]
            bg_inds = bg_inds[I]
            # number of bg in the image
            bg_rois_per_this_gt = rois_per_gt - fg_rois_per_this_gt
            bg_rois_per_this_gt = np.minimum(bg_rois_per_this_gt, bg_inds.size)
            keep_inds.append(bg_inds[0:bg_rois_per_this_gt])

        # only keep unique indexes
        keep_inds = np.unique(keep_inds)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
