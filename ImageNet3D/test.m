function test

K = 10;
opt = globals();

% path = '/home/yuxiang/Projects/mcg';
% if exist(path, 'dir') == 0
%     path = '/scail/scratch/u/yuxiang/mcg';
% end
% tmp = pwd;
% cd(fullfile(path, 'pre-trained'));
% install;
% cd(tmp);

% ImageNet3D paths
root_dir = opt.path_imagenet3d;
image_dir = fullfile(root_dir, 'Images');
image_set_dir = fullfile(root_dir, 'Image_sets');

% read ids
ids = textread(fullfile(image_set_dir, 'test.txt'), '%s');
N = numel(ids);

% main loop
for i = 8217
    filename = sprintf('%s/%s.JPEG', image_dir, ids{i});
    I = imread(filename);
    boxes = selective_search_boxes(I);
    
%     [candidates_scg, ucm2_scg] = im2mcg(I,'fast');
%     boxes = [candidates_scg.bboxes candidates_scg.bboxes_scores];    
    
    fprintf('%d \\ %d, %d boxes\n', i, N, size(boxes, 1));
    
    figure(2);
    imshow(I);
    title(sprintf('%d: selective search', i));
    hold on;
    for j = 1:min(K, size(boxes,1))
        bb = [boxes(j,2) boxes(j,1) boxes(j,4)-boxes(j,2) boxes(j,3)-boxes(j,1)];
        rectangle('Position', bb, 'EdgeColor', 'r', 'LineWidth', 2);
        text(bb(1), bb(2), num2str(j), 'BackgroundColor',[.7 .9 .7]);
    end
    hold off;    
end