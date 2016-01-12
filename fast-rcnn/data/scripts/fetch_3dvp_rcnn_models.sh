#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

FILE=3DVP_RCNN_models.zip
URL=ftp://cs.stanford.edu/cs/cvgl/$FILE

echo "Downloading 3DVP RCNN KITTI models (228M)..."

wget $URL -O $FILE

echo "Unzipping..."

unzip $FILE
