#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/imagenet3d_vgg16_rcnn_view_selective_search.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/VGG16/imagenet3d/solver_rcnn_view.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb imagenet3d_trainval \
  --cfg experiments/cfgs/imagenet3d_rcnn_view_selective_search.yml \
  --iters 160000

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/imagenet3d/test_rcnn_view.prototxt \
  --net output/imagenet3d/imagenet3d_trainval/vgg16_fast_rcnn_view_imagenet3d_selective_search_iter_160000.caffemodel \
  --imdb imagenet3d_test \
  --cfg experiments/cfgs/imagenet3d_rcnn_view_selective_search.yml
