#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/pascal2007_caffenet_rcnn.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/pascal2007/solver_rcnn.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/pascal_rcnn.yml

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/pascal2007/test_rcnn.prototxt \
  --net output/pascal2007/voc_2007_trainval/caffenet_fast_rcnn_pascal2007_iter_40000.caffemodel \
  --imdb voc_2007_test \
  --cfg experiments/cfgs/pascal_rcnn.yml
