# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)

    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0), dtype=np.float32)
    sublabels_blob = np.zeros((0), dtype=np.float32)
    bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
    bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
    if cfg.TRAIN.VIEWPOINT or cfg.TEST.VIEWPOINT:
        view_targets_blob = np.zeros((0, 3 * num_classes), dtype=np.float32)
        view_inside_blob = np.zeros(view_targets_blob.shape, dtype=np.float32)

    # all_overlaps = []
    for im_i in xrange(num_images):
        if cfg.TRAIN.VIEWPOINT or cfg.TEST.VIEWPOINT:
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights, sublabels, view_targets, view_inside_weights \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image, num_classes)
        else:
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights, sublabels \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image, num_classes)

        # Add to RoIs blob
        batch_ind = im_i * np.ones((im_rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, im_rois))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))

        # Add to labels, bbox targets, and bbox loss blobs
        labels_blob = np.hstack((labels_blob, labels))
        sublabels_blob = np.hstack((sublabels_blob, sublabels))
        bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
        bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
        if cfg.TRAIN.VIEWPOINT or cfg.TEST.VIEWPOINT:
            view_targets_blob = np.vstack((view_targets_blob, view_targets))
            view_inside_blob = np.vstack((view_inside_blob, view_inside_weights))

        # all_overlaps = np.hstack((all_overlaps, overlaps))

    # process images
    processed_ims = _process_images(roidb)

    # crop image patches using rois
    im_blob = _get_image_blob(processed_ims, rois_blob)

    # For debug visualizations
    # _vis_minibatch(im_blob, labels_blob, all_overlaps, sublabels_blob)

    blobs = {'data': im_blob}
    blobs['labels'] = labels_blob

    if cfg.TRAIN.BBOX_REG:
        blobs['bbox_targets'] = bbox_targets_blob
        blobs['bbox_inside_weights'] = bbox_inside_blob
        blobs['bbox_outside_weights'] = np.array(bbox_inside_blob > 0).astype(np.float32)

    if cfg.TRAIN.SUBCLS:
        blobs['sublabels'] = sublabels_blob

    if cfg.TRAIN.VIEWPOINT or cfg.TEST.VIEWPOINT:
        blobs['view_targets'] = view_targets_blob
        blobs['view_inside_weights'] = view_inside_blob
        blobs['view_outside_weights'] = np.array(view_inside_blob > 0).astype(np.float32)

    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']
    sublabels = roidb['max_subclasses']
    if cfg.TRAIN.VIEWPOINT or cfg.TEST.VIEWPOINT:
        viewpoints = roidb['max_viewpoints']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = []
    for i in xrange(1, num_classes):
        fg_inds.extend(np.where((labels == i) & (overlaps >= cfg.TRAIN.FG_THRESH[i-1]))[0])
    fg_inds = np.array(fg_inds)

    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image,
                             replace=False)

    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = []
    for i in xrange(1, num_classes):
        bg_inds.extend( np.where((labels == i) & (overlaps < cfg.TRAIN.BG_THRESH_HI[i-1]) &
                        (overlaps >= cfg.TRAIN.BG_THRESH_LO[i-1]))[0] )

    if len(bg_inds) < bg_rois_per_this_image:
        for i in xrange(1, num_classes):
            bg_inds.extend( np.where((labels == i) & (overlaps < cfg.TRAIN.BG_THRESH_HI[i-1]))[0] )

    if len(bg_inds) < bg_rois_per_this_image:
        bg_inds.extend( np.where(overlaps < cfg.TRAIN.BG_THRESH_HI[i-1])[0] )
    bg_inds = np.array(bg_inds, dtype=np.int32)

    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image,
                             replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds).astype(int)
    # print '{} foregrounds and {} backgrounds'.format(fg_inds.size, bg_inds.size)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]
    sublabels = sublabels[keep_inds]
    sublabels[fg_rois_per_this_image:] = 0

    bbox_targets, bbox_loss_weights = \
            _get_bbox_regression_labels(roidb['bbox_targets'][keep_inds, :],
                                        num_classes)

    if cfg.TRAIN.VIEWPOINT or cfg.TEST.VIEWPOINT:
        viewpoints = viewpoints[keep_inds]
        view_targets, view_loss_weights = \
                _get_viewpoint_estimation_labels(viewpoints, labels, num_classes)
        return labels, overlaps, rois, bbox_targets, bbox_loss_weights, sublabels, view_targets, view_loss_weights

    return labels, overlaps, rois, bbox_targets, bbox_loss_weights, sublabels


def _process_images(roidb):
    """Builds an input blob from the images in the roidb
    """
    num_images = len(roidb)
    processed_ims = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        processed_ims.append(im_orig)

    return processed_ims


def _get_image_blob(processed_ims, rois):
    """Builds an input blob from the images in the roidb
    """
    num_rois = rois.shape[0]
    blob = np.zeros((num_rois, 224, 224, 3), dtype=np.float32)

    for i in xrange(num_rois):
        index = int(rois[i, 0])
        im = processed_ims[index]
        height = im.shape[0]
        width = im.shape[1]

        x1 = max(np.floor(rois[i, 1]), 1)
        y1 = max(np.floor(rois[i, 2]), 1)
        x2 = min(np.ceil(rois[i, 3]), width)
        y2 = min(np.ceil(rois[i, 4]), height)

        # crop image
        im_crop = im[y1:y2, x1:x2, :]

        # resize the cropped image
        blob[i, :, :, :] = cv2.resize(im_crop, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)

    return blob


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

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


def _get_viewpoint_estimation_labels(viewpoint_data, clss, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        view_target_data (ndarray): N x 3K blob of regression targets
        view_loss_weights (ndarray): N x 3K blob of loss weights
    """
    view_targets = np.zeros((clss.size, 3 * num_classes), dtype=np.float32)
    view_loss_weights = np.zeros(view_targets.shape, dtype=np.float32)
    inds = np.where( (clss > 0) & np.isfinite(viewpoint_data[:,0]) & np.isfinite(viewpoint_data[:,1]) & np.isfinite(viewpoint_data[:,2]) )[0]
    for ind in inds:
        cls = clss[ind]
        start = 3 * cls
        end = start + 3
        view_targets[ind, start:end] = viewpoint_data[ind, :]
        view_loss_weights[ind, start:end] = [1., 1., 1.]

    assert not np.isinf(view_targets).any(), 'viewpoint undefined'
    return view_targets, view_loss_weights


def _vis_minibatch(im_blob, labels_blob, overlaps, sublabels_blob):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    import math
    for i in xrange(min(im_blob.shape[0], 1)):
        im = im_blob[i, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        subcls = sublabels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' subclass: ', subcls, ' overlap: ', overlaps[i]

        plt.show()
