function selective_search_demo

opt = globals();

% KITTI paths
root_dir = opt.path_kitti_root;
data_set = 'training';

% get sub-directories
cam = 2; % 2 = left color camera
image_dir = fullfile(root_dir,[data_set '/image_' num2str(cam)]);

% get number of images for this dataset
nimages = length(dir(fullfile(image_dir, '*.png')));

% main loop
figure(1);
for img_idx = 0:nimages-1
    I = imread(sprintf('%s/%06d.png', image_dir, img_idx));
    boxes = selective_search_boxes(I);
    
    % show the boxes
    imshow(I);
    hold on;
    for i = 1:size(boxes,1)
        box = boxes(i,:);
        box_draw = [box(2) box(1) box(4)-box(2) box(3)-box(1)];
        rectangle('Position', box_draw, 'EdgeColor', 'g', 'LineWidth', 2);
    end
    pause;    
end