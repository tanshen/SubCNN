# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from utils.cython_nms import nms, nms_new
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os
import math

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    processed_ims = []
    im_scale_factors = []
    scales = cfg.TEST.SCALES_BASE

    for im_scale in scales:
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_boxes_grid(image_height, image_width):
    """
    Return the boxes on image grid.
    """

    # height and width of the heatmap
    height = np.round((image_height * max(cfg.TRAIN.SCALES) - 1) / 4.0 + 1)
    height = np.floor((height - 1) / 2 + 1 + 0.5)
    height = np.floor((height - 1) / 2 + 1 + 0.5)

    width = np.round((image_width * max(cfg.TRAIN.SCALES) - 1) / 4.0 + 1)
    width = np.floor((width - 1) / 2.0 + 1 + 0.5)
    width = np.floor((width - 1) / 2.0 + 1 + 0.5)

    # compute the grid box centers
    h = np.arange(height)
    w = np.arange(width)
    y, x = np.meshgrid(h, w, indexing='ij') 
    centers = np.dstack((x, y))
    centers = np.reshape(centers, (-1, 2))
    num = centers.shape[0]

    # compute width and height of grid box
    area = cfg.TRAIN.KERNEL_SIZE * cfg.TRAIN.KERNEL_SIZE
    aspect = cfg.TRAIN.ASPECTS  # height / width
    num_aspect = len(aspect)
    widths = np.zeros((1, num_aspect), dtype=np.float32)
    heights = np.zeros((1, num_aspect), dtype=np.float32)
    for i in xrange(num_aspect):
        widths[0,i] = math.sqrt(area / aspect[i])
        heights[0,i] = widths[0,i] * aspect[i]

    # construct grid boxes
    centers = np.repeat(centers, num_aspect, axis=0)
    widths = np.tile(widths, num).transpose()
    heights = np.tile(heights, num).transpose()

    x1 = np.reshape(centers[:,0], (-1, 1)) - widths * 0.5
    x2 = np.reshape(centers[:,0], (-1, 1)) + widths * 0.5
    y1 = np.reshape(centers[:,1], (-1, 1)) - heights * 0.5
    y2 = np.reshape(centers[:,1], (-1, 1)) + heights * 0.5
    
    boxes_grid = np.hstack((x1, y1, x2, y2)) / cfg.TRAIN.SPATIAL_SCALE

    return boxes_grid

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)
    scales = np.array(scales)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    if cfg.IS_RPN:
        blobs = {'data' : None, 'boxes_grid' : None}
        blobs['data'], im_scale_factors = _get_image_blob(im)
        blobs['boxes_grid'] = rois
    else:
        blobs = {'data' : None, 'rois' : None}
        blobs['data'], im_scale_factors = _get_image_blob(im)
        blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES_BASE)

    return blobs, im_scale_factors

def _bbox_pred(boxes, box_deltas):
    """Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + cfg.EPS
    heights = boxes[:, 3] - boxes[:, 1] + cfg.EPS
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""

    for i in xrange(boxes.shape[0]):
        boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

    return boxes


def im_detect(net, im, boxes, num_classes, num_subclasses):
    """Detect object classes in an image given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    if boxes.shape[0] == 0:
        scores = np.zeros((0, num_classes))
        pred_boxes = np.zeros((0, 4*num_classes))
        scores_subcls = np.zeros((0, num_subclasses))
        return scores, pred_boxes, scores_subcls

    blobs, unused_im_scale_factors = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['rois'].reshape(*(blobs['rois'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            rois=blobs['rois'].astype(np.float32, copy=False))
    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.SUBCLS:
        scores_subcls = blobs_out['subcls_prob']
    else:
        # just use class scores
        scores_subcls = scores

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = _bbox_pred(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        scores_subcls = scores_subcls[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes, scores_subcls


def im_detect_proposal(net, im, boxes_grid, num_classes, num_subclasses):
    """Detect object classes in an image given boxes on grids.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of boxes

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    blobs, im_scale_factors = _get_blobs(im, boxes_grid)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['boxes_grid'].reshape(*(blobs['boxes_grid'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            boxes_grid=blobs['boxes_grid'].astype(np.float32, copy=False))

    scores_subcls = blobs_out['subcls_prob']
    print scores_subcls.shape

    # build max_scores
    tmp = np.reshape(scores_subcls, (scores_subcls.shape[0], scores_subcls.shape[1]))
    max_scores = np.zeros((scores_subcls.shape[0], num_classes))
    max_scores[:,0] = tmp[:,0]
    max_scores[:,1] = tmp[:,1:].max(axis = 1)
    scores = max_scores

    rois = net.blobs['rois_sub'].data
    inds = rois[:,0]
    boxes = rois[:,1:]
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = _bbox_pred(boxes, box_deltas)
        pred_boxes = _rescale_boxes(pred_boxes, inds, cfg.TRAIN.SCALES)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        pred_boxes = _rescale_boxes(pred_boxes, inds, cfg.TRAIN.SCALES)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)

    # only select one aspect with the highest score
    """
    num = boxes.shape[0]
    num_aspect = len(cfg.TEST.ASPECTS)
    inds = []
    for i in xrange(num/num_aspect):
        index = range(i*num_aspect, (i+1)*num_aspect)
        max_scores = scores[index,1:].max(axis = 1)
        ind_max = np.argmax(max_scores)
        inds.append(index[ind_max])
    """

    # select boxes
    inds = np.where(max_scores[:,1] > cfg.TEST.ROI_THRESHOLD)[0]
    scores = scores[inds]
    pred_boxes = pred_boxes[inds]
    scores_subcls = scores_subcls[inds]
    print scores.shape
   
    # draw boxes
    if 0:
        # print scores, pred_boxes.shape
        import matplotlib.pyplot as plt
        plt.imshow(im)
        for j in xrange(pred_boxes.shape[0]):
            roi = pred_boxes[j,4:]
            plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                           roi[3] - roi[1], fill=False,
                           edgecolor='g', linewidth=3))
        plt.show()

    return scores, pred_boxes, scores_subcls

def vis_detections(im, class_name, dets, thresh=0.1):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    plt.cla()
    plt.imshow(im)
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -2]
        if score > thresh:
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            # plt.title('{}  {:.3f}'.format(class_name, score))
    plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue

            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            inds = np.where((x2 > x1) & (y2 > y1))[0]
            dets = dets[inds,:]
            if dets == []:
                continue

            keep = nms_new(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb):

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    det_file = os.path.join(output_dir, 'detections.pkl')
    if os.path.exists(det_file):
        with open(det_file, 'rb') as fid:
            all_boxes = cPickle.load(fid)
        print 'Detections loaded from {}'.format(det_file)

        if cfg.IS_RPN:
            print 'Evaluating detections'
            imdb.evaluate_proposals(all_boxes, output_dir)
        else:
            print 'Applying NMS to all detections'
            nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)
            print 'Evaluating detections'
            imdb.evaluate_detections(nms_dets, output_dir)
        return

    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = np.inf
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 10000
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(imdb.num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if cfg.IS_RPN == False:
        roidb = imdb.roidb

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        _t['im_detect'].tic()
        if cfg.IS_RPN:
            boxes_grid = _get_boxes_grid(im.shape[0], im.shape[1])
            scores, boxes, scores_subcls = im_detect_proposal(net, im, boxes_grid, imdb.num_classes, imdb.num_subclasses)
        else:
            scores, boxes, scores_subcls = im_detect(net, im, roidb[i]['boxes'], imdb.num_classes, imdb.num_subclasses)
        _t['im_detect'].toc()

        _t['misc'].tic()
        count = 0
        for j in xrange(1, imdb.num_classes):
            if cfg.IS_RPN:
                inds = np.where(scores[:, j] > thresh[j])[0]
            else:
                inds = np.where((scores[:, j] > thresh[j]) & (roidb[i]['gt_classes'] == 0))[0]

            cls_scores = scores[inds, j]
            subcls_scores = scores_subcls[inds, 1:]
            cls_boxes = boxes[inds, j*4:(j+1)*4]

            top_inds = np.argsort(-cls_scores)[:max_per_image]
            cls_scores = cls_scores[top_inds]
            subcls_scores = subcls_scores[top_inds, :]
            cls_boxes = cls_boxes[top_inds, :]
            # push new scores onto the minheap
            for val in cls_scores:
                heapq.heappush(top_scores[j], val)
            # if we've collected more than the max number of detection,
            # then pop items off the minheap and update the class threshold
            if len(top_scores[j]) > max_per_set:
                while len(top_scores[j]) > max_per_set:
                    heapq.heappop(top_scores[j])
                thresh[j] = top_scores[j][0]

            sub_classes = np.reshape(subcls_scores.argmax(axis=1), -1)

            all_boxes[j][i] = \
                    np.hstack((cls_boxes, cls_scores[:, np.newaxis], sub_classes[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
            count = count + len(cls_scores)

            if 0:
                keep = nms_new(all_boxes[j][i], cfg.TEST.NMS)
                vis_detections(im, imdb.classes[j], all_boxes[j][i][keep, :])
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:d} object detected {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, count, _t['im_detect'].average_time, _t['misc'].average_time)

    for j in xrange(1, imdb.num_classes):
        for i in xrange(num_images):
            inds = np.where(all_boxes[j][i][:, 4] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if cfg.IS_RPN:
        print 'Evaluating detections'
        imdb.evaluate_proposals(all_boxes, output_dir)
    else:
        print 'Applying NMS to all detections'
        nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)
        print 'Evaluating detections'
        imdb.evaluate_detections(nms_dets, output_dir)
