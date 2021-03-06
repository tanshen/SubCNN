#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/nthu_caffenet_rcnn_multiscale_6k8k.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_test/test_rcnn_multiscale.prototxt \
  --net data/3DVP_RCNN_models/caffenet_fast_rcnn_multiscale_6k8k_kitti_iter_80000.caffemodel \
  --imdb nthu_370 \
  --cfg experiments/cfgs/nthu_multiscale_6k8k.yml
