#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/nissan_caffenet_combined.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

image_set="autonomy_log_2016-04-11-12-15-46"

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_test/test_rpn.prototxt \
  --net output/kitti/kitti_trainval/caffenet_fast_rcnn_rpn_6k8k_kitti_iter_80000.caffemodel \
  --imdb nissan_$image_set \
  --cfg experiments/cfgs/nissan_rpn.yml

# create an symbol link for the region proposal results
ln -s $PWD/output/nissan/nissan_$image_set/caffenet_fast_rcnn_rpn_6k8k_kitti_iter_80000 data/NISSAN/region_proposals/RPN_6k8k/$image_set

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/kitti_test/test_rcnn_multiscale.prototxt \
  --net output/kitti/kitti_trainval/vgg16_fast_rcnn_multiscale_6k8k_kitti_iter_80000.caffemodel \
  --imdb nissan_$image_set \
  --cfg experiments/cfgs/nissan_multiscale_6k8k.yml
