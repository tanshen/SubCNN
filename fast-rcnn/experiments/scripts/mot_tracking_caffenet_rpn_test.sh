#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/mot_tracking_caffenet_rpn_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

image_set="train"

mot_train_seqs=("TUD-Stadtmitte" "TUD-Campus" "PETS09-S2L1" \
            "ETH-Bahnhof" "ETH-Sunnyday" "ETH-Pedcross2" "ADL-Rundle-6" \
            "ADL-Rundle-8" "KITTI-13" "KITTI-17" "Venice-2")

for i in "${mot_train_seqs[@]}"
do

echo $i

if [ -f $PWD/output/mot/mot_tracking_$image_set\_$i/caffenet_fast_rcnn_rpn_mot_iter_80000/detections.pkl ]
then
  rm $PWD/output/mot/mot_tracking_$image_set\_$i/caffenet_fast_rcnn_rpn_mot_iter_80000/detections.pkl
fi

if [ -h data/MOT_Tracking/region_proposals_trainval/$image_set/$i ]
then
  rm data/MOT_Tracking/region_proposals_trainval/$image_set/$i
fi

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/mot_tracking_train/test_rpn.prototxt \
  --net output/mot/mot_tracking_train_trainval/caffenet_fast_rcnn_rpn_mot_iter_80000.caffemodel \
  --imdb mot_tracking_$image_set\_$i \
  --cfg experiments/cfgs/mot_rpn.yml

# create an symbol link for the region proposal results
ln -s $PWD/output/mot/mot_tracking_$image_set\_$i/caffenet_fast_rcnn_rpn_mot_iter_80000 data/MOT_Tracking/region_proposals_trainval/$image_set/$i

done


image_set="test"

mot_test_seqs=("TUD-Crossing" "PETS09-S2L2" "ETH-Jelmoli" \
            "ETH-Linthescher" "ETH-Crossing" "AVG-TownCentre" "ADL-Rundle-1" \
            "ADL-Rundle-3" "KITTI-16" "KITTI-19" "Venice-1")

for i in "${mot_test_seqs[@]}"
do

echo $i

if [ -f $PWD/output/mot/mot_tracking_$image_set\_$i/caffenet_fast_rcnn_rpn_mot_iter_80000/detections.pkl ]
then
  rm $PWD/output/mot/mot_tracking_$image_set\_$i/caffenet_fast_rcnn_rpn_mot_iter_80000/detections.pkl
fi

if [ -h data/MOT_Tracking/region_proposals_trainval/$image_set/$i ]
then
  rm data/MOT_Tracking/region_proposals_trainval/$image_set/$i
fi

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/mot_tracking_train/test_rpn.prototxt \
  --net output/mot/mot_tracking_train_trainval/caffenet_fast_rcnn_rpn_mot_iter_80000.caffemodel \
  --imdb mot_tracking_$image_set\_$i \
  --cfg experiments/cfgs/mot_rpn.yml

# create an symbol link for the region proposal results
ln -s $PWD/output/mot/mot_tracking_$image_set\_$i/caffenet_fast_rcnn_rpn_mot_iter_80000 data/MOT_Tracking/region_proposals_trainval/$image_set/$i

done
