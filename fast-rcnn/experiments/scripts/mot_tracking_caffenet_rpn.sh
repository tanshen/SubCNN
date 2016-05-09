#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/mot_tracking_caffenet_rpn.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/mot_tracking_train/solver_rpn.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb mot_tracking_train_train \
  --cfg experiments/cfgs/mot_rpn.yml \
  --iters 40000
