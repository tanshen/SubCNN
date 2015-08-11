#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import matplotlib
matplotlib.use('GTK')
import matplotlib.pyplot as plt
import _init_paths
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np
import Image
from fast_rcnn.config import cfg, cfg_from_file
import cv2

if __name__ == '__main__':
    print('Using config:')
    pprint.pprint(cfg)

    prototxt = 'models/CaffeNet/kitti_val/roi.prototxt'
    pretrain = 'data/imagenet_models/CaffeNet.v2.caffemodel'
    image_path = 'data/demo/000004.jpg'

    caffe.set_mode_gpu()
    caffe.set_device(3)

    # initialize net
    net = caffe.Net(prototxt, caffe.TEST)
    net.copy_from(pretrain)

    ksize = net.params['conv_sub'][0].data.shape[2:]
    # make Gaussian blur
    sigma = 1.
    y, x = np.mgrid[-ksize[0]//2 + 1:ksize[0]//2 + 1, -ksize[1]//2 + 1:ksize[1]//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    gaussian = (g / g.sum()).astype(np.float32)
    net.params['conv_sub'][0].data[0] = gaussian

    # read image
    im = cv2.imread(image_path)
    plt.title("original image")
    plt.imshow(im)
    plt.axis('off')

    # input data
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_input = im_orig[np.newaxis, :, :, :]
    channel_swap = (0, 3, 1, 2)
    im_input = im_input.transpose(channel_swap)
    print im_input.shape

    net.blobs['data'].reshape(*im_input.shape)
    net.blobs['data'].data[...] = im_input

    net.forward()

    # pick filter output
    conv_sub = net.blobs['conv_sub'].data[0, 0]
    print conv_sub.shape

    # draw bounding boxes on the original image
    kernel_size = 3
    scale = 16
    value = conv_sub.mean()
    print value

    h = np.arange(conv_sub.shape[0])
    w = np.arange(conv_sub.shape[1])
    y, x = np.meshgrid(h, w, indexing='ij')
    
    scores = np.reshape(conv_sub[y, x], -1)
    print scores.shape

    tmp = np.dstack((x, y))
    tmp = np.reshape(tmp, (-1, 2))
    boxes = np.hstack((tmp - (kernel_size-1) * np.ones(tmp.shape) / 2, tmp + (kernel_size-1) * np.ones(tmp.shape) / 2))
    print boxes.shape
    print boxes

    for i in xrange(boxes.shape[0]):
        if scores[i] > value:
           cx = (boxes[i,0] + boxes[i,2]) / 2 * scale
           cy = (boxes[i,1] + boxes[i,3]) / 2 * scale
           height = (boxes[i,3] - boxes[i,1] + 1) * scale
           width = (boxes[i,2] - boxes[i,0] + 1) * scale
           plt.gca().add_patch(plt.Rectangle((cx, cy), width, height, fill=False, edgecolor='g', linewidth=3))

    """
    for x in xrange(conv_sub.shape[1]):
        for y in xrange(conv_sub.shape[0]):
            if conv_sub[y,x] > value:
                cx = x*scale
                cy = y*scale
                plt.gca().add_patch(plt.Rectangle((cx, cy), width, height, fill=False, edgecolor='g', linewidth=3))
    """
   
    plt.show()
