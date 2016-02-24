function selective_search_ImageNet3D

matlabpool open;

opt = globals();

% ImageNet3D paths
root_dir = opt.path_imagenet3d;
image_dir = fullfile(root_dir, 'Images');
image_set_dir = fullfile(root_dir, 'Image_sets');

% output dir
out_dir = 'region_proposals/selective_search';
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
    boxes = selective_search_boxes(I);
    fprintf('%d \\ %d, %d boxes\n', i, N, size(boxes, 1));
    % write results
    fid = fopen(sprintf('%s/%s.txt', out_dir, ids{i}), 'w');
    for j = 1:size(boxes,1)
        fprintf(fid, '%d %d %d %d\n', boxes(j,1), boxes(j,2), boxes(j,3), boxes(j,4));
    end
    fclose(fid);
end

matlabpool close;