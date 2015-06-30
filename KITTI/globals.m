function opt = globals()

% KITTI paths
path_kitti = '/net/acadia/workplace/yuxiang/Projects/KITTI';
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

% add selective search path
addpath(genpath('3rd_party/SelectiveSearchCodeIJCV'));