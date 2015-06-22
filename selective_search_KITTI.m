function selective_search_KITTI

matlabpool open;

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
    boxes{i} = selective_search_boxes(I);
    images{i} = sprintf('%06d', img_idx);
    fprintf('%d \\ %d, %d boxes\n', i, N, size(boxes{i}, 1));
end

filename = sprintf('KITTI_%s.mat', data_set);
save(filename, 'boxes', 'images', '-v7.3');

matlabpool close;