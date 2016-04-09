#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/kitti_test_vgg16_rcnn_multiscale_6k8k.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/VGG16/kitti_test/solver_rcnn_multiscale_6k8k.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb kitti_trainval \
  --cfg experiments/cfgs/kitti_rcnn_multiscale_6k8k.yml \
  --iters 80000

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/kitti_test/test_rcnn_multiscale.prototxt \
  --net output/kitti/kitti_trainval/vgg16_fast_rcnn_multiscale_6k8k_kitti_iter_80000.caffemodel \
  --imdb kitti_test \
  --cfg experiments/cfgs/kitti_rcnn_multiscale_6k8k.yml
