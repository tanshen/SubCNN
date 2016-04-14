#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/kitti_tracking_test_caffenet_rpn_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

image_set="training"

for i in {0..20}
do

seq_num=$(printf '%04d' "$i")
echo $seq_num

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_tracking_test/test_rpn.prototxt \
  --net output/kitti/kitti_tracking_training_trainval/caffenet_fast_rcnn_rpn_trainval_kitti_iter_40000.caffemodel \
  --imdb kitti_tracking_$image_set\_$seq_num \
  --cfg experiments/cfgs/kitti_tracking_rpn.yml

# create an symbol link for the region proposal results
ln -s $PWD/output/kitti_tracking/kitti_tracking_$image_set\_$seq_num/caffenet_fast_rcnn_rpn_trainval_kitti_iter_40000 data/KITTI_Tracking/region_proposals/RPN_trainval/$image_set/$seq_num

done


image_set="testing"

for i in {0..28}
do

seq_num=$(printf '%04d' "$i")
echo $seq_num

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_tracking_test/test_rpn.prototxt \
  --net output/kitti/kitti_tracking_training_trainval/caffenet_fast_rcnn_rpn_trainval_kitti_iter_40000.caffemodel \
  --imdb kitti_tracking_$image_set\_$seq_num \
  --cfg experiments/cfgs/kitti_tracking_rpn.yml

# create an symbol link for the region proposal results
ln -s $PWD/output/kitti_tracking/kitti_tracking_$image_set\_$seq_num/caffenet_fast_rcnn_rpn_trainval_kitti_iter_40000 data/KITTI_Tracking/region_proposals/RPN_trainval/$image_set/$seq_num

done
