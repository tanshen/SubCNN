#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/pascal3d_vgg16_rpn_msr.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/VGG16/pascal3d/solver_rpn_msr.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb pascal3d_train \
  --cfg experiments/cfgs/pascal3d_rpn_msr_vgg16.yml \
  --iters 40000

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/pascal3d/test_rpn_msr.prototxt \
  --net output/pascal3d/pascal3d_train/vgg16_fast_rcnn_rpn_msr_pascal3d_iter_40000.caffemodel \
  --imdb pascal3d_val \
  --cfg experiments/cfgs/pascal3d_rpn_msr_vgg16.yml

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/pascal3d/test_rpn_msr.prototxt \
  --net output/pascal3d/pascal3d_train/vgg16_fast_rcnn_rpn_msr_pascal3d_iter_40000.caffemodel \
  --imdb pascal3d_train \
  --cfg experiments/cfgs/pascal3d_rpn_msr_vgg16.yml

