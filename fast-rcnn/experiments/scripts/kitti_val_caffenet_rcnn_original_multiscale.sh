#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/kitti_val_caffenet_rcnn_original_multiscale.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/kitti_val/solver_rcnn_original_multiscale.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb kitti_train \
  --cfg experiments/cfgs/kitti_rcnn_original_multiscale.yml

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_val/test_rcnn_original_multiscale.prototxt \
  --net output/kitti/kitti_train/caffenet_fast_rcnn_original_multiscale_kitti_iter_40000.caffemodel \
  --imdb kitti_val \
  --cfg experiments/cfgs/kitti_rcnn_original_multiscale.yml
