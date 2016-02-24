function opt = globals()

% KITTI paths
path_kitti = '/net/acadia0a/data/yxiang/KITTI_Dataset';
if exist(path_kitti, 'dir') == 0
    path_kitti = '/home/yuxiang/Projects/KITTI_Dataset';
end
if exist(path_kitti, 'dir') == 0
    path_kitti = '/scail/scratch/u/yuxiang/KITTI_Dataset';
end

opt.path_kitti = path_kitti;
opt.path_kitti_devkit = [opt.path_kitti '/devkit/matlab'];
opt.path_kitti_root = [opt.path_kitti '/data_object_image_2'];

% add kitti devit path
addpath(opt.path_kitti_devkit);

% ImageNet3D paths
path_imagenet3d = '/capri5/Projects/ImageNet3D';
if exist(path_imagenet3d, 'dir') == 0
    path_imagenet3d = '/scail/scratch/u/yuxiang/ImageNet3D';
end
opt.path_imagenet3d = path_imagenet3d;

% add selective search path
addpath(genpath('../3rd_party/SelectiveSearchCodeIJCV'));
