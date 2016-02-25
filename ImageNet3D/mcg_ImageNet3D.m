function mcg_ImageNet3D

matlabpool open;

path = '/home/yuxiang/Projects/mcg';
if exist(path, 'dir') == 0
    path = '/scail/scratch/u/yuxiang/mcg';
end
tmp = pwd;
cd(fullfile(path, 'pre-trained'));
install;
cd(tmp);

opt = globals();

% ImageNet3D paths
root_dir = opt.path_imagenet3d;
image_dir = fullfile(root_dir, 'Images');
image_set_dir = fullfile(root_dir, 'Image_sets');

% output dir
out_dir = 'region_proposals/mcg';
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
    [candidates_scg, ucm2_scg] = im2mcg(I,'fast');
    boxes = [candidates_scg.bboxes candidates_scg.bboxes_scores];
    fprintf('%d \\ %d, %d boxes\n', i, N, size(boxes, 1));
    
    % write results
    fid = fopen(sprintf('%s/%s.txt', out_dir, ids{i}), 'w');
    for j = 1:size(boxes,1)
        fprintf(fid, '%d %d %d %d %f\n', boxes(j,1), boxes(j,2), boxes(j,3), boxes(j,4), boxes(j,5));
    end
    fclose(fid);    
end

matlabpool close;