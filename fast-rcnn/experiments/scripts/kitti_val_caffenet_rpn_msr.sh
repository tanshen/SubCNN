#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/kitti_val_caffenet_rpn_msr.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/kitti_val/solver_rpn_msr.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb kitti_train \
  --cfg experiments/cfgs/kitti_rpn_msr.yml \
  --iters 40000

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_val/test_rpn_msr.prototxt \
  --net output/kitti/kitti_train/caffenet_fast_rcnn_rpn_msr_kitti_iter_20000.caffemodel \
  --imdb kitti_val \
  --cfg experiments/cfgs/kitti_rpn_msr.yml

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_val/test_rpn_msr.prototxt \
  --net output/kitti/kitti_train/caffenet_fast_rcnn_rpn_msr_kitti_iter_40000.caffemodel \
  --imdb kitti_train \
  --cfg experiments/cfgs/kitti_rpn_msr.yml
