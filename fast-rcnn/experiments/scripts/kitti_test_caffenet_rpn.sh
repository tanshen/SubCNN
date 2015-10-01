#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/kitti_test_caffenet_rpn.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#time ./tools/train_net.py --gpu $1 \
#  --solver models/CaffeNet/kitti_test/solver_rpn.prototxt \
#  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
#  --imdb kitti_trainval \
#  --cfg experiments/cfgs/kitti_rpn.yml \
#  --iters 40000

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_test/test_rpn.prototxt \
  --net output/kitti/kitti_trainval/caffenet_fast_rcnn_rpn_kitti_iter_40000.caffemodel \
  --imdb kitti_trainval \
  --cfg experiments/cfgs/kitti_rpn.yml

#time ./tools/test_net.py --gpu $1 \
#  --def models/CaffeNet/kitti_test/test_rpn.prototxt \
#  --net output/kitti/kitti_trainval/caffenet_fast_rcnn_rpn_kitti_iter_40000.caffemodel \
#  --imdb kitti_test \
#  --cfg experiments/cfgs/kitti_rpn.yml
