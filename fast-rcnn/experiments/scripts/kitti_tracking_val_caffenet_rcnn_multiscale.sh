#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/kitti_tracking_val_caffenet_rcnn_multiscale.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/kitti_tracking_val/solver_rcnn_multiscale.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb kitti_tracking_training_train \
  --cfg experiments/cfgs/kitti_tracking_multiscale_train.yml

image_set="training"

for i in {0..20}
do

seq_num=$(printf '%04d' "$i")
echo $seq_num

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_tracking_val/test_rcnn_multiscale.prototxt \
  --net output/kitti/kitti_tracking_training_train/caffenet_fast_rcnn_multiscale_kitti_iter_40000.caffemodel \
  --imdb kitti_tracking_$image_set\_$seq_num \
  --cfg experiments/cfgs/kitti_tracking_multiscale_train.yml

# copy the detection result
cp output/kitti_tracking/kitti_tracking_$image_set\_$seq_num/caffenet_fast_rcnn_multiscale_kitti_iter_40000/$seq_num.txt data/KITTI_Tracking/detection_train/$image_set

done
