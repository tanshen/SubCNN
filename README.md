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
        author = {Xiang, Yu and Choi, Wongun and Lin, Yuanqing and Savarese, Silvio},
        title = {Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection},
        booktitle = {arXiv:1604.04693},
        year = {2016}
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
1. Download the KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php).

2. Create symlinks for the KITTI detection dataset
    ```Shell
    cd $ROOT/fast-rcnn/data/KITTI
    ln -s $data_object_image_2 data_object_image_2
    ```

3. Unzip the voxel_exemplars.zip in $ROOT/fast-rcnn/data/KITTI

4. Run the region proposal network to generate region proposals
    ```Shell
    cd $ROOT/fast-rcnn

    # subcategory-aware RPN for validation
    ./experiments/scripts/kitti_val_caffenet_rpn.sh $GPU_ID

    # subcategory-aware RPN for testing
    ./experiments/scripts/kitti_test_caffenet_rpn_6k8k.sh $GPU_ID

    # Faster RCNN RPN for validation
    ./experiments/scripts/kitti_val_caffenet_rpn_msr.sh $GPU_ID

    # Faster RCNN RPN for testing
    ./experiments/scripts/kitti_test_caffenet_rpn_msr_6k8k.sh $GPU_ID

    ```

5. Copy the region proposals to $ROOT/fast-rcnn/data/KITTI/region_proposals/RPN_*:
    ```Shell
    # validation (125 subcategories for car)
    $ROOT/fast-rcnn/data/KITTI/region_proposals/RPN_125/training   # a directory contains region proposals for training images: 000000.txt, ..., 007480.txt

    # testing (227 subcategories for car)
    $ROOT/fast-rcnn/data/KITTI/region_proposals/RPN_227/training   # a directory contains region proposals for training images: 000000.txt, ..., 007480.txt
    $ROOT/fast-rcnn/data/KITTI/region_proposals/RPN_227/testing    # a directory contains region proposals for testing  images: 000000.txt, ..., 007517.txt
    ```

5. Run the detection network
    ```Shell
    cd $ROOT/fast-rcnn

    # subcategory-aware detection network for validation
    ./experiments/scripts/kitti_val_caffenet_rcnn_multiscale.sh $GPU_ID

    # subcategory-aware detection network for testing
    ./experiments/scripts/kitti_test_caffenet_rcnn_multiscale_6k8k.sh $GPU_ID

    # subcategory-aware detection network for testing with VGG16
    ./experiments/scripts/kitti_test_vgg16_rcnn_multiscale_6k8k.sh $GPU_ID

    # subcategory-aware detection network for testing with GoogleNet
    ./experiments/scripts/kitti_test_googlenet_rcnn.sh $GPU_ID

    # Faster RCNN detection network for validation
    ./experiments/scripts/kitti_val_caffenet_rcnn_msr.sh $GPU_ID

    # Faster RCNN detection network for testing
    ./experiments/scripts/kitti_test_caffenet_rcnn_original_msr.sh $GPU_ID

    ```

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
