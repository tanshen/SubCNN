#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/kitti_val_google_rcnn.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/GoogleNet/kitti_val/solver_rcnn.prototxt \
  --weights data/imagenet_models/bvlc_googlenet.caffemodel \
  --imdb kitti_train \
  --cfg experiments/cfgs/kitti_rcnn_googlenet.yml \
  --iters 40000

time ./tools/test_net.py --gpu $1 \
  --def models/GoogleNet/kitti_val/test_rcnn.prototxt \
  --net output/kitti/kitti_train/googlenet_fast_rcnn_kitti_iter_40000.caffemodel \
  --imdb kitti_val \
  --cfg experiments/cfgs/kitti_rcnn_googlenet.yml
