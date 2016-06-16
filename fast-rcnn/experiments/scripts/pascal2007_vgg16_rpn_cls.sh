#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/pascal2007_vgg16_rpn_cls.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/VGG16/pascal2007/solver_rpn_cls.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/pascal_rpn_cls_vgg16.yml \
  --iters 40000

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/pascal2007/test_rpn_cls.prototxt \
  --net output/pascal2007/voc_2007_trainval/vgg16_fast_rcnn_rpn_cls_pascal2007_iter_40000.caffemodel \
  --imdb voc_2007_test \
  --cfg experiments/cfgs/pascal_rpn_cls_vgg16.yml

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/pascal2007/test_rpn_cls.prototxt \
  --net output/pascal2007/voc_2007_trainval/vgg16_fast_rcnn_rpn_cls_pascal2007_iter_40000.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/pascal_rpn_cls_vgg16.yml

