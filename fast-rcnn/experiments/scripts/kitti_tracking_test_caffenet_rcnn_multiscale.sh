#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/kitti_tracking_test_caffenet_rcnn_multiscale.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/kitti_tracking_test/solver_rcnn_multiscale.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb kitti_tracking_training_trainval \
  --cfg experiments/cfgs/kitti_tracking_multiscale_train.yml \
  --iters 80000

image_set="testing"

for i in {0..28}
do

seq_num=$(printf '%04d' "$i")
echo $seq_num

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_tracking_test/test_rcnn_multiscale.prototxt \
  --net output/kitti_tracking/kitti_tracking_training_trainval/caffenet_fast_rcnn_multiscale_trainval_kitti_iter_80000.caffemodel \
  --imdb kitti_tracking_$image_set\_$seq_num \
  --cfg experiments/cfgs/kitti_tracking_multiscale_train.yml

# copy the detection result
cp output/kitti_tracking/kitti_tracking_$image_set\_$seq_num/caffenet_fast_rcnn_multiscale_trainval_kitti_iter_80000/$seq_num.txt data/KITTI_Tracking/detection_trainval/$image_set

done
