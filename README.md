# SubCNN: Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection

Created by Yu Xiang at CVGL, Stanford University,
based on the Fast R-CNN created by Ross Girshick at Microsoft Research, Redmond.

### Introduction

We introduce a new region proposal network that uses subcategory information to guide the proposal generating process, and a new detection network for joint detection and subcategory classification. By using subcategories related to object pose, we achieve state-of-the-art performance on both detection and pose estimation on commonly used benchmarks, such as KITTI and PASCAL3D+.

This package supports
 - Subcategory-aware region proposal network
 - Subcategory-aware detection network
 - Region proposal network in Faster R-CNN (Ren et al. NIPS 2015)
 - Detection network in Faster R-CNN (Ren et al. NIPS 2015)
 - Experiments on the following datasets: KITTI Detection, PASCAL VOC, PASCAL3D+, KITTI Tracking sequences, MOT sequences

### License

SubCNN is released under the MIT License (refer to the LICENSE file for details).

### Citing Fast R-CNN

If you find SubCNN useful in your research, please consider citing:

@incollection{xiang2016subcategory,
  author    = {Xiang, Yu and Choi, Wongun and Lin, Yuanqing and Savarese, Silvio},
  title     = {Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection},
  booktitle = {arXiv:1604.04693},
  year      = {2016}
}

### Installation

1. Clone the SubCNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/yuxng/SubCNN.git
  ```
  
2. We'll call the directory that you cloned SubCNN into `ROOT`

   *Ignore notes 1 and 2 if you followed step 1 above.*
   
   **Note 1:** If you didn't clone SubCNN with the `--recursive` flag, then you'll need to manually clone the `caffe-fast-rcnn` submodule:
    ```Shell
    git submodule update --init --recursive
    ```
    **Note 2:** The `caffe-fast-rcnn` submodule needs to be on the `fast-rcnn` branch (or equivalent detached state). This will happen automatically *if you follow these instructions*.

3. Build the Cython modules
    ```Shell
    cd $ROOT/fast-rcnn/lib
    make
    ```
    
4. Build Caffe and pycaffe
    ```Shell
    cd $ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```
    
5. Download pre-computed 3DVP R-CNN detectors trained on KITTI
    ```Shell
    cd $ROOT/fast-rcnn
    ./data/scripts/fetch_3dvp_rcnn_models.sh
    ```

    This will populate the `$ROOT/fast-rcnn/data` folder with `3DVP_RCNN_models`.

### Running with the KITTI detection dataset

### Running with the NTHU dataset
1. The NTHU dataset should have a directory named 'data', under which it has the following structure:
    ```Shell
  	$data/                           # the directory contains all the data
  	$data/71                         # a directory for video 71: 000001.jpg, ..., 002956.jpg
  	$data/71.txt                     # a txt file contains the frame names: 000001 \n 000002 \n ... 002956
  	# ... and several other directories and txt files ...
    ```

2. Create symlinks for the NTHU dataset
    ```Shell
    cd $ROOT/fast-rcnn/data/NTHU
    ln -s $data data
    ```

3. Run the region proposal network to generate region proposals, modify the script to run with different videos
    ```Shell
    cd $ROOT/fast-rcnn
    ./experiments/scripts/nthu_caffenet_rpn_6k8k.sh $GPU_ID
    ```

4. Copy the region proposals to $ROOT/fast-rcnn/data/NTHU/region_proposals/RPN_6k8k:
    ```Shell
    $ROOT/fast-rcnn/data/NTHU/region_proposals/RPN_6k8k/71    # a directory contains region proposals for video 71: 000001.txt, ..., 002956.txt
    ```

5. Run the detection network, modify the script to run with different videos
    ```Shell
    cd $ROOT/fast-rcnn
    ./experiments/scripts/nthu_caffenet_rcnn_multiscale_6k8k.sh $GPU_ID
    ```
