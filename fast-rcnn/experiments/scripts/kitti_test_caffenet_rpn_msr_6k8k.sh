#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/kitti_test_caffenet_rpn_msr_6k8k.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/kitti_test/solver_rpn_msr_6k8k.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb kitti_trainval \
  --cfg experiments/cfgs/kitti_rpn_msr.yml \
  --iters 80000

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_test/test_rpn_msr.prototxt \
  --net output/kitti/kitti_trainval/caffenet_fast_rcnn_rpn_msr_6k8k_kitti_iter_80000.caffemodel \
  --imdb kitti_trainval \
  --cfg experiments/cfgs/kitti_rpn_msr.yml

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_test/test_rpn_msr.prototxt \
  --net output/kitti/kitti_trainval/caffenet_fast_rcnn_rpn_msr_6k8k_kitti_iter_40000.caffemodel \
  --imdb kitti_test \
  --cfg experiments/cfgs/kitti_rpn_msr.yml
