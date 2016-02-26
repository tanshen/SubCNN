#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/imagenet3d_caffenet_rcnn_original_selective_search.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/imagenet3d/solver_rcnn_original.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb imagenet3d_trainval \
  --cfg experiments/cfgs/imagenet3d_rcnn_original_selective_search.yml \
  --iters 40000

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/imagenet3d/test_rcnn_original.prototxt \
  --net output/imagenet3d/imagenet3d_trainval/caffenet_fast_rcnn_original_imagenet3d_selective_search_iter_40000.caffemodel \
  --imdb imagenet3d_test \
  --cfg experiments/cfgs/imagenet3d_rcnn_original_selective_search.yml
