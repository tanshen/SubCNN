#!/bin/bash

set -e
GPU=$1
./experiments/scripts/kitti_val_caffenet1.sh $GPU
./experiments/scripts/kitti_val_caffenet2.sh $GPU
./experiments/scripts/kitti_val_caffenet3.sh $GPU
./experiments/scripts/kitti_val_caffenet4.sh $GPU
./experiments/scripts/kitti_val_caffenet5.sh $GPU
