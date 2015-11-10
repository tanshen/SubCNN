function opt = globals()

opt.root = '/home/yuxiang/Projects/PASCAL3D+_release1.1';
opt.path_pascal = '/home/yuxiang/Projects/PASCAL3D+_release1.1/PASCAL/VOCdevkit';

SLMpaths = {'/net/skyserver30/workplace/local/wongun/yuxiang/SLM', ...
    '/net/skyserver10/workplace/yxiang/SLM', ...
    '/home/yuxiang/Projects/SLM', ...
    '/scail/scratch/u/yuxiang/SLM'};

for i = 1:numel(SLMpaths)
    if exist(SLMpaths{i}, 'dir')
        opt.SLMroot = SLMpaths{i};
        break;
    end
end