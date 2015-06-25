__author__ = 'wongun' # derived from honda.py by fyang

import os
import datasets.imdb
import numpy as np
import scipy.sparse
import utils.cython_bbox
import subprocess
import re
import cPickle
from collections import defaultdict

class kitti(datasets.imdb):
    def __init__(self, image_set, data_path=None):
        datasets.imdb.__init__(self, 'kitti_' + image_set)
        self._image_set = image_set
        self._data_path = data_path
        # self._classes = ('__background__', 'Car', 'Person', 'Bike', 'Motorbike', 'Truck', 'Bus')
        self._classes = ('__background__', 'Car', 'Pedestrian')
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._class_to_ind['Van'] = 1
        self._class_to_ind['Person_sitting'] = 2
        self._class_to_ind['Cyclist'] = 2
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'top_k': 100000}

        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'images',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._data_path, 'ExpList', self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.rstrip('\n') for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        # sub_dir = ['innercity', 'innercity2', 'innercity3']
        gt_boxes = defaultdict(list)
        gt_classes = defaultdict(list)
        gt_overlaps = defaultdict(list)
        gt_dontcare = defaultdict(list)
        print 'Loading ground truth boxes...'
        tol_gt = 0

        ann_dir = os.path.join(self._data_path, 'Annotations')
        annotation_files = ['train.txt']
        for af in annotation_files:
            with open(os.path.join(ann_dir, af)) as f:
                lines = f.readlines()
                for line in lines:
                    ov = [0] * self.num_classes
                    words = line.split()
                    boxes = [int(float(n)) for n in words[1:5]]
                    cls = words[9].strip('"')

                    imname = 'train/'+words[5]

                    if 'Dont' in cls:
                        gt_dontcare[imname].append(boxes)
                    elif cls in self._class_to_ind:
                        cls_id = self._class_to_ind[cls]
                        gt_classes[imname].append(cls_id)
                        # ov[1] = 1.0
                        ov[cls_id] = 1.0
                        gt_overlaps[imname].append(ov)
                        gt_boxes[imname].append(boxes)

        gt_roidb = []
        keys = ['boxes', 'gt_classes', 'gt_overlaps', 'flipped', 'gt_dontcare']
        to_keep = []
        for i in xrange(len(self.image_index)):
            if np.array(gt_boxes[self.image_index[i]]).shape[0] > 0:
                gt_roidb.append({keys[0]: np.array(gt_boxes[self.image_index[i]]),
                                keys[1]: np.array(gt_classes[self.image_index[i]]),
                                keys[2]: scipy.sparse.csr_matrix(np.array(gt_overlaps[self.image_index[i]])),
                                keys[3]: False,
                                keys[4]: np.array(gt_dontcare[self.image_index[i]])})
                to_keep.append(i)
                tol_gt += np.array(gt_boxes[self.image_index[i]]).shape[0]

        print tol_gt
        print self.num_images
        self._image_index = list(np.array(self._image_index)[to_keep])
        print self.num_images

        print 'Ground truth boxes loaded'
        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        if self._image_set != 'yxiang_val':
            gt_roidb = self.gt_roidb()
            print 'Loading selective search boxes...'
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            print 'Selective search boxes loaded'

            print 'Loading ACF boxes...'
            acf_roidb = self._load_acf_roidb(gt_roidb)
            print 'ACF boxes loaded'
            roidb = datasets.imdb.merge_roidbs(ss_roidb, gt_roidb)
            roidb = datasets.imdb.merge_roidbs(roidb, acf_roidb)
        else:
            print 'Loading selective search boxes...'
            roidb = self._load_selective_search_roidb(None)
            print 'Selective search boxes loaded'
            print 'Loading ACF boxes...'
            acf_roidb = self._load_acf_roidb(None)
            print 'ACF boxes loaded'
            roidb = datasets.imdb.merge_roidbs(roidb, acf_roidb)
        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_' + self._image_set + '_selective_search_raw_box_list.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                box_list = cPickle.load(fid)
            print '{} ss raw boxes loaded from {}'.format(self.name, cache_file)
        else:
            box_list = []
            for img in self.image_index:
                filename = os.path.join(self._data_path, 'CandBB/ss/', img + '.txt')
                assert os.path.exists(filename), \
                   'Selective search data not found at: {}'.format(filename)
                raw_data = np.loadtxt(filename, dtype=int)
                box_list.append(raw_data[:min(self.config['top_k'], raw_data.shape[0]), 1:])
            with open(cache_file, 'wb') as fid:
                cPickle.dump(box_list, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote ss raw boxes to {}'.format(cache_file)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_acf_roidb(self, gt_roidb):
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_' + self._image_set + '_acf_raw_box_list.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                box_list = cPickle.load(fid)
            print '{} acf raw boxes loaded from {}'.format(self.name, cache_file)
        else:
            box_list = []
            for img in self.image_index:
                filename = os.path.join(self._data_path, 'ACFBB', 'car', img + '.txt')
                assert os.path.exists(filename), \
                   'ACF data not found at: {}'.format(filename)
                raw_data = np.loadtxt(filename, usecols=(2,3,4,5), dtype=float).astype(int)
                box_list.append(raw_data[:min(self.config['top_k'], raw_data.shape[0]), :])
            with open(cache_file, 'wb') as fid:
                cPickle.dump(box_list, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote acf raw boxes to {}'.format(cache_file)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

if __name__ == '__main__':
    d = datasets.kitti('/net/acadia3/data/RegionletSetup/KITTI', 'yxiang_train')
    res = d.roidb
    from IPython import embed; embed()
