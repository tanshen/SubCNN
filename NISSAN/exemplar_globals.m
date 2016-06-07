% Set up global variables used throughout the code

% directory with KITTI development kit and dataset
KITTIpaths = {'/net/skyserver10/workplace/yxiang/KITTI_Dataset', ...
    '/net/acadia/workplace/yuxiang/Projects/KITTI', ...
    '/capri5/Projects/KITTI_Dataset', ...
    '/scratch/yuxiang/Projects/KITTI_Dataset', ...
    '/scail/scratch/u/yuxiang/KITTI_Dataset'};

for i = 1:numel(KITTIpaths)
    if exist(KITTIpaths{i}, 'dir')
        KITTIroot = [KITTIpaths{i} '/data_object_image_2'];
        KITTIdevkit = [KITTIpaths{i} '/devkit/matlab'];
        break;
    end
end

addpath(KITTIdevkit);

SLMpaths = {'/net/skyserver30/workplace/local/wongun/yuxiang/SLM', ...
    '/net/skyserver10/workplace/yxiang/SLM', ...
    '/capri5/Projects/SLM', ...
    '/scail/scratch/u/yuxiang/SLM'};

for i = 1:numel(SLMpaths)
    if exist(SLMpaths{i}, 'dir')
        SLMroot = SLMpaths{i};
        break;
    end
end

PASCAL3Dpaths = {'/capri5/Projects/PASCAL3D+_release1.1', ...
    '/capri5/Projects/Pose_Dataset/PASCAL3D+_release1.1', ...
    '/scratch/yuxiang/Projects/PASCAL3D+_release1.1', ...
    '/scail/scratch/u/yuxiang/PASCAL3D+_release1.1'};

for i = 1:numel(PASCAL3Dpaths)
    if exist(PASCAL3Dpaths{i}, 'dir')
        PASCAL3Droot = PASCAL3Dpaths{i};
        path_pascal = [PASCAL3Droot '/PASCAL/VOCdevkit'];
        path_img_imagenet = [PASCAL3Droot '/Images/%s_imagenet'];
        path_cad = [PASCAL3Droot '/CAD/%s.mat'];
        break;
    end
end

% pascal init
tmp = pwd;
cd(path_pascal);
addpath([cd '/VOCcode']);
VOCinit;
cd(tmp);