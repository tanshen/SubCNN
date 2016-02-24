function edgeboxes_ImageNet3D

matlabpool open;

path = '/home/yuxiang/Projects/edges';
if exist(path, 'dir') == 0
    path = '/scail/scratch/u/yuxiang/edges';
end
addpath(genpath(path));

path_toolbox = '/home/yuxiang/Projects/SLM/3rd_party/piotr_toolbox';
if exist(path_toolbox, 'dir') == 0
    path_toolbox = '/scail/scratch/u/yuxiang/SLM/3rd_party/piotr_toolbox';
end
addpath(genpath(path_toolbox));

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

% ImageNet3D paths
root_dir = opt.path_imagenet3d;
image_dir = fullfile(root_dir, 'Images');
image_set_dir = fullfile(root_dir, 'Image_sets');

% output dir
out_dir = 'region_proposals/edge_boxes';
if exist(out_dir, 'dir') == 0
    mkdir(out_dir);
end

% read ids
ids = textread(fullfile(image_set_dir, 'all.txt'), '%s');
N = numel(ids);

% main loop
parfor i = 1:N
    filename = sprintf('%s/%s.JPEG', image_dir, ids{i});
    I = imread(filename);
    if numel(size(I)) == 2
        I = repmat(I,[1 1 3]);
    end
    boxes = edgeBoxes(I,model,opts);
    fprintf('%d \\ %d, %d boxes\n', i, N, size(boxes, 1));
    
    % write results
    fid = fopen(sprintf('%s/%s.txt', out_dir, ids{i}), 'w');
    for j = 1:size(boxes,1)
        fprintf(fid, '%d %d %d %d %f\n', boxes(j,1), boxes(j,2), boxes(j,3), boxes(j,4), boxes(j,5));
    end
    fclose(fid);    
end

matlabpool close;
