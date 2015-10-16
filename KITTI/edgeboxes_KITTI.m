function edgeboxes_KITTI

matlabpool open;

path = '/home/yuxiang/Projects/edges';
addpath(genpath(path));

addpath(genpath('/home/yuxiang/Projects/SLM/3rd_party/piotr_toolbox'));

% load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load([path '/models/forest/modelBsds']);
model = model.model;
model.opts.multiscale=0;
model.opts.sharpen=2;
model.opts.nThreads=4;

% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 2000;  % max number of boxes to detect

opt = globals();

% KITTI paths
root_dir = opt.path_kitti_root;
data_set = 'training';

% get sub-directories
cam = 2; % 2 = left color camera
image_dir = fullfile(root_dir,[data_set '/image_' num2str(cam)]);

% get number of images for this dataset
N = length(dir(fullfile(image_dir, '*.png')));
boxes = cell(1, N);
images = cell(1, N);

% main loop
parfor i = 1:N
    img_idx = i - 1;
    filename = sprintf('%s/%06d.png', image_dir, img_idx);
    I = imread(filename);
    boxes{i} = edgeBoxes(I,model,opts);
    images{i} = sprintf('%06d', img_idx);
    fprintf('%d \\ %d, %d boxes\n', i, N, size(boxes{i}, 1));
end

filename = sprintf('KITTI_%s_edgeboxes.mat', data_set);
save(filename, 'boxes', 'images', '-v7.3');

matlabpool close;