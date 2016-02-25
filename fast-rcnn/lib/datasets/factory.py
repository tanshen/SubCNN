# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.pascal_voc
import datasets.imagenet3d
import datasets.kitti
import datasets.kitti_tracking
import numpy as np

def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = datasets.pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb
"""
# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))

# Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
# but only returning the first k boxes
for top_k in np.arange(1000, 11000, 1000):
    for year in ['2007', '2012']:
        for split in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}_top_{:d}'.format(year, split, top_k)
            __sets[name] = (lambda split=split, year=year, top_k=top_k:
                    _selective_search_IJCV_top_k(split, year, top_k))
"""

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        print name
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))

# KITTI dataset
for split in ['train', 'val', 'trainval', 'test']:
    name = 'kitti_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.kitti(split))

# PASCAL3D+ dataset
for split in ['train', 'val']:
    name = 'pascal3d_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.pascal3d(split))

# ImageNet3D dataset
for split in ['train', 'val', 'trainval', 'test', 'test_1', 'test_2']:
    name = 'imagenet3d_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.imagenet3d(split))

# NISSAN dataset
for split in ['2015-10-21-16-25-12', '2016-01-15-15-05-24', '2016-02-17-16-51-05', '2016-02-17-16-55-12', '2016-02-17-16-59-25']:
    name = 'nissan_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.nissan(split))

# NTHU dataset
for split in ['71', '370']:
    name = 'nthu_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.nthu(split))

# KITTI Tracking dataset
for split in ['0000', '0001', '0002', '0003', '0004', '0005', \
    '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', \
    '0015', '0016', '0017', '0018', '0019', '0020']:
    name = 'kitti_tracking_{}_{}'.format('training', split)
    print name
    __sets[name] = (lambda split=split:
            datasets.kitti_tracking('training', split))

for split in ['0000', '0001', '0002', '0003', '0004', '0005', \
    '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', \
    '0015', '0016', '0017', '0018', '0019', '0020', '0021', '0022', \
    '0023', '0024', '0025', '0026', '0027', '0028']:
    name = 'kitti_tracking_{}_{}'.format('testing', split)
    print name
    __sets[name] = (lambda split=split:
            datasets.kitti_tracking('testing', split))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
