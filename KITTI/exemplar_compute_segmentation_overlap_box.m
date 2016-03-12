function exemplar_compute_segmentation_overlap_box

is_save = 1;

% KITTI path
exemplar_globals;
root_dir = KITTIroot;
data_set = 'training';
cam = 2;
image_dir = fullfile(root_dir, [data_set '/image_' num2str(cam)]);

% read ids of validation images
object = load('kitti_ids_new.mat');
ids = object.ids_val;
M = numel(ids);

% read detection results
result_dir = 'results_faster_rcnn';
detections = cell(1, M);
for i = 1:M
    filename = sprintf('%s/%06d.txt', result_dir, ids(i));
    disp(filename);
    fid = fopen(filename, 'r');
    C = textscan(fid, '%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f');   
    fclose(fid);
    
    det = double([C{5} C{6} C{7} C{8} C{end}]);
    index = strcmp('Car', C{1});
    det = det(index, :);
    detections{i} = det;
end
fprintf('load detection done\n');

% load data
object = load(fullfile(SLMroot, 'KITTI/data.mat'));
data = object.data;

% for each image
for i = 1:M
    img_idx = ids(i);
    disp(img_idx);
    
    % read ground truth
    filename = sprintf('%s/KITTI/Annotations/%06d.mat', SLMroot, img_idx);
    object = load(filename);
    gt = object.record.objects;
    imgsize = object.record.imgsize;
    
    % build the pattern for ground truth
    num = numel(gt);
    fprintf('%d gts\n', num);
    Patterns_gt = uint8(zeros(imgsize(2), imgsize(1), num));
    Boxes_gt = zeros(num, 4);
    for k = 1:num
        if strcmp('Car', gt(k).type) == 1 && isempty(gt(k).pattern) == 0
            bbox_pr = [gt(k).x1 gt(k).y1 gt(k).x2 gt(k).y2];
            bbox = zeros(1,4);
            bbox(1) = max(1, floor(bbox_pr(1)));
            bbox(2) = max(1, floor(bbox_pr(2)));
            bbox(3) = min(imgsize(1), floor(bbox_pr(3)));
            bbox(4) = min(imgsize(2), floor(bbox_pr(4)));
            w = bbox(3) - bbox(1) + 1;
            h = bbox(4) - bbox(2) + 1;
            
            Boxes_gt(k,:) = bbox;

            % apply the 2D occlusion mask to the bounding box
            % check if truncated pattern
            pattern = gt(k).pattern;                
            index = find(pattern == 1);
            if gt(k).truncation > 0 && isempty(index) == 0                
                [y, x] = ind2sub(size(pattern), index);
                cx = size(pattern, 2)/2;
                cy = size(pattern, 1)/2;
                width = size(pattern, 2);
                height = size(pattern, 1);                 
                pattern = pattern(min(y):max(y), min(x):max(x));

                % find the object center
                sx = w / size(pattern, 2);
                sy = h / size(pattern, 1);
                tx = bbox(1) - sx*min(x);
                ty = bbox(2) - sy*min(y);
                cx = sx * cx + tx;
                cy = sy * cy + ty;
                width = sx * width;
                height = sy * height;
                bbox_pr = round([cx-width/2 cy-height/2 cx+width/2 cy+height/2]);
                width = bbox_pr(3) - bbox_pr(1) + 1;
                height = bbox_pr(4) - bbox_pr(2) + 1;
                
                pattern = imresize(gt(k).pattern, [height width], 'nearest');
                
                bbox = zeros(1,4);
                bbox(1) = max(1, floor(bbox_pr(1)));
                start_x = bbox(1) - floor(bbox_pr(1)) + 1;
                bbox(2) = max(1, floor(bbox_pr(2)));
                start_y = bbox(2) - floor(bbox_pr(2)) + 1;
                bbox(3) = min(imgsize(1), floor(bbox_pr(3)));
                bbox(4) = min(imgsize(2), floor(bbox_pr(4)));
                w = bbox(3) - bbox(1) + 1;
                h = bbox(4) - bbox(2) + 1;
                pattern = pattern(start_y:start_y+h-1, start_x:start_x+w-1);
            else
                pattern = imresize(pattern, [h w], 'nearest');
            end
            
            % build the pattern in the image
            height = imgsize(2);
            width = imgsize(1);
            P = uint8(zeros(height, width));
            x = bbox(1);
            y = bbox(2);
            index_y = y:min(y+h-1, height);
            index_x = x:min(x+w-1, width);
            P(index_y, index_x) = pattern(1:numel(index_y), 1:numel(index_x));
            Patterns_gt(:,:,k) = P;
        else
            Boxes_gt(k,:) = [0 0 -1 -1];
        end
    end
    
    % get predicted bounding box
    det = detections{i};
    
    if is_save == 0
        pick = nms_new(det, 0.6);
        det = det(pick, :);
        pick = det(:,6) > -3;
        det = det(pick, :);
    end
    
    % build the patterns for detections
    num = size(det, 1);
    fprintf('%d detections\n', num);
    Patterns_box = uint8(zeros(imgsize(2), imgsize(1), num));
    Boxes_det = zeros(num, 4);
    for k = 1:num
        bbox_pr = det(k,1:4);
        bbox = zeros(1,4);
        bbox(1) = max(1, floor(bbox_pr(1)));
        bbox(2) = max(1, floor(bbox_pr(2)));
        bbox(3) = min(imgsize(1), floor(bbox_pr(3)));
        bbox(4) = min(imgsize(2), floor(bbox_pr(4)));
        w = bbox(3) - bbox(1) + 1;
        h = bbox(4) - bbox(2) + 1;
        
        % build the box and its pattern
        Boxes_det(k,:) = bbox;
        height = imgsize(2);
        width = imgsize(1);
        P = uint8(zeros(height, width));
        x = bbox(1);
        y = bbox(2);
        index_y = y:min(y+h-1, height);
        index_x = x:min(x+w-1, width);
        P(index_y, index_x) = 1;
        Patterns_box(:,:,k) = P;
    end
    
    % compute matching score between detections and groundtruths
    Scores_box = compute_matching_scores_segmentation(Boxes_det, Patterns_box, Boxes_gt, Patterns_gt);
    
    if is_save == 0
        file_img = sprintf('%s/%06d.png', image_dir, img_idx);
        I = imread(file_img);
        imshow(I);
        
        for k = 1:size(det, 1);
            % get predicted bounding box
            bbox_pr = det(k,1:4);
            bbox_draw = [bbox_pr(1), bbox_pr(2), bbox_pr(3)-bbox_pr(1), bbox_pr(4)-bbox_pr(2)];
            rectangle('Position', bbox_draw, 'EdgeColor', 'g', 'LineWidth', 2);
            s = sprintf('%.2f:%.2f', max(Scores(k,:)), max(Scores_box(k,:)));
            text(bbox_pr(1), bbox_pr(2), s, 'FontSize', 8, 'BackgroundColor', 'c');
        end
        pause;
    else
        filename = fullfile(result_dir, sprintf('%06d_seg.mat', i));
        save(filename, 'Scores_box');
    end
end