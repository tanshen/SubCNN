# 3DVP_RCNN

### Installation

1. Clone the 3DVP_RCNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/yuxng/3DVP_RCNN.git
  ```
  
2. We'll call the directory that you cloned 3DVP_RCNN into `ROOT`

   *Ignore notes 1 and 2 if you followed step 1 above.*
   
   **Note 1:** If you didn't clone 3DVP_RCNN with the `--recursive` flag, then you'll need to manually clone the `caffe-fast-rcnn` submodule:
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
    
5. Download pre-computed 3DVP R-CNN detectors on KITTI
    ```Shell
    cd $ROOT/fast-rcnn
    ./data/scripts/fetch_3dvp_rcnn_models.sh
    ```

    This will populate the `$ROOT/fast-rcnn/data` folder with `3DVP_RCNN_models`.

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
    cd $ROOT/3DVP_RCNN/fast-rcnn/data/NTHU
    ln -s $data data
    ```

3. Run the region proposal network to generate region proposals
    ```Shell
    cd $ROOT/3DVP_RCNN/fast-rcnn
    ./experiments/scripts/nthu_caffenet_rpn_6k8k.sh $GPU_ID
    ```
