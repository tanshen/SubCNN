function recall_all = compute_recall_simple

classes = {'car', 'pedestrian', 'cyclist'};
MIN_OVERLAPS = [0.7, 0.5, 0.5];

% evaluation parameter
MIN_HEIGHT = [40, 25, 25];     % minimum height for evaluated groundtruth/detections
MAX_OCCLUSION = [0, 1, 2];     % maximum occlusion level of the groundtruth used for evaluation
MAX_TRUNCATION = [0.15, 0.3, 0.5]; % maximum truncation level of the groundtruth used for evaluation

% KITTI path
opt = globals();
root_dir = opt.path_kitti_root;
data_set = 'training';
cam = 2;
label_dir = fullfile(root_dir, [data_set '/label_' num2str(cam)]);

% read ids of validation images
object = load('kitti_ids_new.mat');
ids = object.ids_val;
% ids = sort([object.ids_train, object.ids_val]);
M = numel(ids);

% read ground truth
groundtruths = cell(1, M);
for i = 1:M
    % read ground truth 
    img_idx = ids(i);
    groundtruths{i} = readLabels(label_dir, img_idx);
end
fprintf('load ground truth done\n');

% read detection results
% object = load('KITTI_training');
% boxes = object.boxes;
% detections = cell(1, M);
% for i = 1:M
%     ind = ids(i) + 1;
%     det = [boxes{ind}(:,2) boxes{ind}(:,1) boxes{ind}(:,4) boxes{ind}(:,3)];
%     detections{i} = det;
% end

% result_dir = 'kitti_train_ap_125';
% filename = sprintf('/home/yuxiang/Projects/SLM/ACF/%s/car_3d_aps_125_combined_test.mat', result_dir);
% object = load(filename);
% detections = object.dets;
% count = 0;
% for i = 1:M
%     count = count + size(detections{i}, 1);
% end

count = 0;
detections = cell(1, M);
for i = 1:M
    filename = sprintf('region_proposals/%06d.txt', ids(i));
    disp(filename);
    fid = fopen(filename, 'r');
    C = textscan(fid, '%f %f %f %f %f');   
    fclose(fid);
    
    det = double([C{1} C{2} C{3} C{4} C{5}]);
    ind = (det(:,3) > det(:,1)) & (det(:,4) > det(:,2)) & (det(:,5) > 0.01);
    det = det(ind,1:4);

    detections{i} = det;
    count = count + size(detections{i}, 1);
end

fprintf('load detection done\n');
fprintf('%f detections per image\n', count / M);
    
for k = 1:numel(classes)
    cls = classes{k};
    MIN_OVERLAP = MIN_OVERLAPS(k);    

    recall_all = cell(1, 3);

    for difficulty = 1:3
        % for each image
        ignored_gt_all = cell(1, M);
        dontcare_gt_all = cell(1, M);
        for i = 1:M
            gt = groundtruths{i};
            num = numel(gt);
            % clean data
            % extract ground truth bounding boxes for current evaluation class
            ignored_gt = zeros(1, num);
            n_gt = 0;
            dontcare_gt = zeros(1, num);
            n_dc = 0;
            for j = 1:num
                if strcmpi(cls, gt(j).type) == 1
                    valid_class = 1;
                elseif strcmpi('van', gt(j).type) == 1
                    valid_class = 0;
                else
                    valid_class = -1;
                end

                height = gt(j).y2 - gt(j).y1;    
                if(gt(j).occlusion > MAX_OCCLUSION(difficulty) || ...
                    gt(j).truncation > MAX_TRUNCATION(difficulty) || ...
                    height < MIN_HEIGHT(difficulty))
                    ignore = true;            
                else
                    ignore = false;
                end

                if valid_class == 1 && ignore == false
                    ignored_gt(j) = 0;
                    n_gt = n_gt + 1;
                elseif valid_class == 0 || (valid_class == 1 && ignore == true) 
                    ignored_gt(j) = 1;
                else
                    ignored_gt(j) = -1;
                end

                if strcmp('DontCare', gt(j).type) == 1
                    dontcare_gt(j) = 1;
                    n_dc = n_dc + 1;
                end
            end

            ignored_gt_all{i} = ignored_gt;
            dontcare_gt_all{i} = dontcare_gt;
        end

        fn = 0;
        tp = 0;

        % for each image
        for i = 1:M
            gt = groundtruths{i};
            num = numel(gt);
            ignored_gt = ignored_gt_all{i};

            det = detections{i};
            det = truncate_detections(det);    
            num_det = size(det, 1);

            % compute statistics
            assigned_detection = zeros(1, num_det);
            % for each ground truth
            for j = 1:num
                if ignored_gt(j) == -1 || ignored_gt(j) == 1
                    continue;
                end

                box_gt = [gt(j).x1 gt(j).y1 gt(j).x2 gt(j).y2];
                if num_det == 0
                    overlap = 0;
                else
                    overlap = boxoverlap(det, box_gt);
                end
                % disp(max(overlap));
                if max(overlap) >= MIN_OVERLAP
                    tp = tp + 1;
                else
                    fn = fn + 1;
                end

            end
        end

        % compute recall and precision
        recall = tp / (tp + fn);

        recall_all{difficulty} = recall;
        fprintf('%s difficulty %d, tp %d, fn %d, recall %f\n', cls, difficulty, tp, fn, recall);
    end
end


function det_new = truncate_detections(det)

if isempty(det) == 0
    imsize = [1224, 370]; % kittisize
    det(det(:, 1) < 0, 1) = 0;
    det(det(:, 2) < 0, 2) = 0;
    det(det(:, 1) > imsize(1), 1) = imsize(1);
    det(det(:, 2) > imsize(2), 2) = imsize(2);
    det(det(:, 3) < 0, 1) = 0;
    det(det(:, 4) < 0, 2) = 0;
    det(det(:, 3) > imsize(1), 3) = imsize(1);
    det(det(:, 4) > imsize(2), 4) = imsize(2);
end
det_new = det;
