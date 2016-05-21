#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/mot_tracking_val_caffenet_rcnn_multiscale.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/mot_tracking_train/solver_rcnn_multiscale.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb mot_tracking_train_train \
  --cfg experiments/cfgs/mot_tracking_multiscale.yml

image_set="train"

mot_train_seqs=("TUD-Stadtmitte" "TUD-Campus" "PETS09-S2L1" \
            "ETH-Bahnhof" "ETH-Sunnyday" "ETH-Pedcross2" "ADL-Rundle-6" \
            "ADL-Rundle-8" "KITTI-13" "KITTI-17" "Venice-2")

for i in "${mot_train_seqs[@]}"
do

echo $i

if [ -f $PWD/output/mot/mot_tracking_$image_set\_$i/caffenet_fast_rcnn_multiscale_mot_iter_40000/detections.pkl ]
then
  rm $PWD/output/mot/mot_tracking_$image_set\_$i/caffenet_fast_rcnn_multiscale_mot_iter_40000/detections.pkl
fi

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/mot_tracking_train/test_rcnn_multiscale.prototxt \
  --net output/mot/mot_tracking_train_train/caffenet_fast_rcnn_multiscale_mot_iter_40000.caffemodel \
  --imdb mot_tracking_$image_set\_$i \
  --cfg experiments/cfgs/mot_tracking_multiscale.yml

# copy the detection result
cp output/mot/mot_tracking_$image_set\_$i/caffenet_fast_rcnn_multiscale_mot_iter_40000/$i.txt data/MOT_Tracking/detection_train/$image_set

done
