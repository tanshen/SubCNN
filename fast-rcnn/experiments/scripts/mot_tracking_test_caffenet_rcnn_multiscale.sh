#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/mot_tracking_test_caffenet_rcnn_multiscale.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/VGG16/mot_tracking_test/solver_rcnn_multiscale.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb mot_tracking_train_trainval \
  --cfg experiments/cfgs/mot_tracking_multiscale_vgg16.yml \
  --iters 80000

image_set="train"

mot_train_seqs=("TUD-Stadtmitte" "TUD-Campus" "PETS09-S2L1" \
            "ETH-Bahnhof" "ETH-Sunnyday" "ETH-Pedcross2" "ADL-Rundle-6" \
            "ADL-Rundle-8" "KITTI-13" "KITTI-17" "Venice-2")

for i in "${mot_train_seqs[@]}"
do

echo $i

if [ -f $PWD/output/mot/mot_tracking_$image_set\_$i/vgg16_fast_rcnn_multiscale_mot_iter_80000/detections.pkl ]
then
  rm $PWD/output/mot/mot_tracking_$image_set\_$i/vgg16_fast_rcnn_multiscale_mot_iter_80000/detections.pkl
fi

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/mot_tracking_test/test_rcnn_multiscale.prototxt \
  --net output/mot/mot_tracking_train_trainval/vgg16_fast_rcnn_multiscale_mot_iter_80000.caffemodel \
  --imdb mot_tracking_$image_set\_$i \
  --cfg experiments/cfgs/mot_tracking_multiscale_vgg16.yml

# copy the detection result
cp output/mot/mot_tracking_$image_set\_$i/vgg16_fast_rcnn_multiscale_mot_iter_80000/$i.txt data/MOT_Tracking/detection_trainval/$image_set

done


image_set="test"

mot_test_seqs=("TUD-Crossing" "PETS09-S2L2" "ETH-Jelmoli" \
            "ETH-Linthescher" "ETH-Crossing" "AVG-TownCentre" "ADL-Rundle-1" \
            "ADL-Rundle-3" "KITTI-16" "KITTI-19" "Venice-1")

for i in "${mot_test_seqs[@]}"
do

echo $i

if [ -f $PWD/output/mot/mot_tracking_$image_set\_$i/vgg16_fast_rcnn_multiscale_mot_iter_80000/detections.pkl ]
then
  rm $PWD/output/mot/mot_tracking_$image_set\_$i/vgg16_fast_rcnn_multiscale_mot_iter_80000/detections.pkl
fi

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/mot_tracking_test/test_rcnn_multiscale.prototxt \
  --net output/mot/mot_tracking_train_trainval/vgg16_fast_rcnn_multiscale_mot_iter_80000.caffemodel \
  --imdb mot_tracking_$image_set\_$i \
  --cfg experiments/cfgs/mot_tracking_multiscale_vgg16.yml

# copy the detection result
cp output/mot/mot_tracking_$image_set\_$i/vgg16_fast_rcnn_multiscale_mot_iter_80000/$i.txt data/MOT_Tracking/detection_trainval/$image_set

done
