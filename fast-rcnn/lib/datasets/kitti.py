__author__ = 'yuxiang' # derived from honda.py by fyang

import datasets
import datasets.kitti
import os
import datasets.imdb
import numpy as np
import scipy.sparse
import utils.cython_bbox
import subprocess
import cPickle

class kitti(datasets.imdb):
    def __init__(self, image_set, kitti_path=None):
        datasets.imdb.__init__(self, 'kitti_' + image_set)
        self._image_set = image_set
        self._kitti_path = self._get_default_path() if kitti_path is None \
                            else kitti_path
        self._data_path = os.path.join(self._kitti_path, 'data_object_image_2')
        self._classes = ('__background__', 'Car')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.voxel_pattern_roidb

        assert os.path.exists(self._kitti_path), \
                'KITTI path does not exist: {}'.format(self._kitti_path)
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
        # set the prefix
        if self._image_set == 'test':
            prefix = 'testing/image_2'
        else:
            prefix = 'training/image_2'

        image_path = os.path.join(self._data_path, prefix, index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._kitti_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.rstrip('\n') for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where KITTI is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'KITTI')


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_kitti_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def _load_kitti_annotation(self, index):
        """
        Load image and bounding boxes info from txt file in the KITTI format.
        """
        filename = os.path.join(self._data_path, 'training', 'label_2', index + '.txt')

        lines = []
        with open(filename) as f:
            for line in f:
                words = line.split()
                cls = words[0]
                if cls in self._class_to_ind:
                    lines.append(line)
            

        num_objs = len(lines)

        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for ix, line in enumerate(lines):
            words = line.split()
            cls = self._class_to_ind[words[0]]
            boxes[ix, :] = [float(n) for n in words[4:8]]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def voxel_pattern_roidb(self):
        """
        Return the database of 3D voxel pattern regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_voxel_pattern_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} vp roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            print 'Loading voxel pattern boxes...'
            vp_roidb = self._load_voxel_pattern_roidb(gt_roidb)
            print 'Voxel pattern boxes loaded'
            roidb = datasets.imdb.merge_roidbs(vp_roidb, gt_roidb)
        else:
            print 'Loading voxel pattern boxes...'
            roidb = self._load_voxel_pattern_roidb(None)
            print 'Voxel pattern boxes loaded'

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote vp roidb to {}'.format(cache_file)

        return roidb

    def _load_voxel_pattern_roidb(self, gt_roidb):
        # set the prefix
        model = '3DVP_125/'
        if self._image_set == 'test':
            prefix = model + 'testing'
        else:
            prefix = model + 'training'

        box_list = []
        for index in self.image_index:
            filename = os.path.join(self._kitti_path, prefix, index + '.txt')
            assert os.path.exists(filename), \
                'Voxel pattern data not found at: {}'.format(filename)
            raw_data = np.loadtxt(filename, dtype=float)
            if len(raw_data.shape) == 1:
                if raw_data.size == 0:
                    raw_data = raw_data.reshape((0, 4))
                else:
                    raw_data = raw_data.reshape((1, 4))
            box_list.append(raw_data)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

if __name__ == '__main__':
    d = datasets.kitti('train')
    res = d.roidb
    from IPython import embed; embed()
