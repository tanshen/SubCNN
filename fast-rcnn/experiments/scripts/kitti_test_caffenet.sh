#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/kitti_test_caffenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/kitti_test/solver.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb kitti_trainval \
  --cfg experiments/cfgs/kitti_5scales.yml \
  --iters 60000

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_test/test.prototxt \
  --net output/kitti/kitti_trainval/caffenet_fast_rcnn_kitti_iter_60000.caffemodel \
  --imdb kitti_test \
  --cfg experiments/cfgs/kitti_5scales.yml
