#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/kitti_val_caffenet_joint.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# first train region proposal network
#time ./tools/train_net.py --gpu $1 \
#  --solver models/CaffeNet/kitti_val/solver_rpn.prototxt \
#  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
#  --imdb kitti_train \
#  --cfg experiments/cfgs/kitti_rpn.yml \
#  --iters 20000

# train the joint network
time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/kitti_val/solver_joint.prototxt \
  --weights output/kitti/kitti_train/caffenet_fast_rcnn_rpn_kitti_iter_20000.caffemodel \
  --imdb kitti_train \
  --cfg experiments/cfgs/kitti_joint.yml \
  --iters 40000

# test the joint network
time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_val/test_joint.prototxt \
  --net output/kitti/kitti_train/caffenet_fast_rcnn_joint_kitti_iter_40000.caffemodel \
  --imdb kitti_val \
  --cfg experiments/cfgs/kitti_joint.yml
