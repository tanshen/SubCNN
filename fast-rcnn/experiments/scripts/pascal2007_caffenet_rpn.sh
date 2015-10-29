#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/pascal2007_caffenet_rpn.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/pascal2007/solver_rpn.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/pascal_rpn.yml \
  --iters 80000

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/pascal2007/test_rpn.prototxt \
  --net output/pascal2007/voc_2007_trainval/caffenet_fast_rcnn_rpn_pascal2007_iter_80000.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/pascal_rpn.yml

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/pascal2007/test_rpn.prototxt \
  --net output/pascal2007/voc_2007_trainval/caffenet_fast_rcnn_rpn_pascal2007_iter_80000.caffemodel \
  --imdb voc_2007_test \
  --cfg experiments/cfgs/pascal_rpn.yml
