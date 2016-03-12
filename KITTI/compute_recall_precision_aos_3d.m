function [fppi_all, thresholds_all] = compute_recall_precision_aos_3d

cls = 'car';

% evaluation parameter
MIN_HEIGHT = [40, 25, 25];     % minimum height for evaluated groundtruth/detections
MAX_OCCLUSION = [0, 1, 2];     % maximum occlusion level of the groundtruth used for evaluation
MAX_TRUNCATION = [0.15, 0.3, 0.5]; % maximum truncation level of the groundtruth used for evaluation
MIN_OVERLAP = 0.7;
N_SAMPLE_PTS = 41;

% KITTI path
exemplar_globals;
root_dir = KITTIroot;
data_set = 'training';
cam = 2;
label_dir = fullfile(root_dir, [data_set '/label_' num2str(cam)]);

% load data
object = load('../KITTI/data.mat');
data = object.data;

% read ids of validation images
object = load('kitti_ids_new.mat');
ids = object.ids_val;
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
result_dir = 'kitti_train_ap_125';
filename = sprintf('%s/odets_3d.mat', result_dir);
object = load(filename);
detections = object.dets_3d;
fprintf('load detection done\n');

% read segmentation scores
segmentations = cell(1, M);
segmentations_box = cell(1, M);
for i = 1:M
    filename = sprintf('%s/%06d_seg.mat', result_dir, i);
    object = load(filename);
    segmentations{i} = object.Scores;
    segmentations_box{i} = object.Scores_box;
end
fprintf('load segmentation scores done\n');

recall_all = cell(1, 3);
precision_all = cell(1, 3);
fppi_all = cell(1, 3);
aos_all = cell(1, 3);
asa_all = cell(1, 3);
asa_box_all = cell(1, 3);
ala_5_all = cell(1, 3);
ala_2_all = cell(1, 3);
ala_1_all = cell(1, 3);
thresholds_all = cell(1, 3);

for difficulty = 1:3
    % for each image
    scores_all = [];
    n_gt_all = 0;
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
        
        % compute statistics
        
        % get predicted bounding box
        det_3d = detections{i};
        n = numel(det_3d);
        det = zeros(n, 9);
        for j = 1:n
            det(j,:) = [det_3d(j).x1 det_3d(j).y1 det_3d(j).x2 det_3d(j).y2 ...
                det_3d(j).alpha det_3d(j).score det_3d(j).t(1) det_3d(j).t(2) det_3d(j).t(3)];
        end
        det = truncate_detections(det);
        
        num_det = size(det, 1);
        assigned_detection = zeros(1, num_det);
        scores = [];
        count = 0;
        for j = 1:num
            if ignored_gt(j) == -1
                continue;
            end
            
            box_gt = [gt(j).x1 gt(j).y1 gt(j).x2 gt(j).y2];
            valid_detection = -inf;
            % find the maximum score for the candidates and get idx of respective detection
            for k = 1:num_det
                if assigned_detection(k) == 1
                    continue;
                end
                overlap = boxoverlap(det(k,:), box_gt);
                if overlap > MIN_OVERLAP && det(k,6) > valid_detection
                    det_idx = k;
                    valid_detection = det(k,6);
                end
            end
            
            if isinf(valid_detection) == 0 && ignored_gt(j) == 1
                assigned_detection(det_idx) = 1;
            elseif isinf(valid_detection) == 0
                assigned_detection(det_idx) = 1;
                count = count + 1;
                scores(count) = det(det_idx, 6);
            end
        end
        scores_all = [scores_all scores];
        n_gt_all = n_gt_all + n_gt;
        ignored_gt_all{i} = ignored_gt;
        dontcare_gt_all{i} = dontcare_gt;
    end
    % get thresholds
    thresholds = get_thresholds(scores_all, n_gt_all, N_SAMPLE_PTS);
    
    nt = numel(thresholds);
    tp = zeros(nt, 1);
    fp = zeros(nt, 1);
    fn = zeros(nt, 1);
    similarity = zeros(nt, 1);
    overlap_seg = zeros(nt, 1);
    overlap_box = zeros(nt, 1);
    accuracy_5 = zeros(nt, 1);
    accuracy_2 = zeros(nt, 1);
    accuracy_1 = zeros(nt, 1);
    recall = zeros(nt, 1);
    precision = zeros(nt, 1);
    fppi = zeros(nt, 1);
    aos = zeros(nt, 1);
    asa = zeros(nt, 1);
    asa_box = zeros(nt, 1);
    ala_5 = zeros(nt, 1);
    ala_2 = zeros(nt, 1);
    ala_1 = zeros(nt, 1);
    
    % for each image
    for i = 1:M
        disp(i);
        gt = groundtruths{i};
        num = numel(gt);
        ignored_gt = ignored_gt_all{i};
        
        % get predicted bounding box
        det_3d = detections{i};
        n = numel(det_3d);
        det = zeros(n, 9);
        for j = 1:n
            det(j,:) = [det_3d(j).x1 det_3d(j).y1 det_3d(j).x2 det_3d(j).y2 ...
                det_3d(j).alpha det_3d(j).score det_3d(j).t(1) det_3d(j).t(2) det_3d(j).t(3)];
        end
        det = truncate_detections(det);        
        
        num_det = size(det, 1);
        
        seg = segmentations{i};
        seg_box = segmentations_box{i};
        if num && num_det
            assert(size(seg,1) == num_det);
            assert(size(seg,2) == num);
            assert(size(seg_box,1) == num_det);
            assert(size(seg_box,2) == num);    
        end
        
        % for each threshold
        for t = 1:nt
            % compute statistics
            assigned_detection = zeros(1, num_det);
            % for each ground truth
            for j = 1:num
                if ignored_gt(j) == -1
                    continue;
                end

                box_gt = [gt(j).x1 gt(j).y1 gt(j).x2 gt(j).y2];
                valid_detection = -inf;
                max_overlap = 0;
                % for computing pr curve values, the candidate with the greatest overlap is considered
                for k = 1:num_det
                    if assigned_detection(k) == 1
                        continue;
                    end
                    if det(k,6) < thresholds(t)
                        continue;
                    end
                    overlap = boxoverlap(det(k,:), box_gt);
                    if overlap > MIN_OVERLAP && overlap > max_overlap
                        max_overlap = overlap;
                        det_idx = k;
                        valid_detection = 1;
                    end
                end

                if isinf(valid_detection) == 1 && ignored_gt(j) == 0
                    fn(t) = fn(t) + 1;
                elseif isinf(valid_detection) == 0 && ignored_gt(j) == 1
                    assigned_detection(det_idx) = 1;
                elseif isinf(valid_detection) == 0
                    tp(t) = tp(t) + 1;
                    assigned_detection(det_idx) = 1;
                    % compute alpha
                    alpha = det(det_idx, 5);
                    delta = gt(j).alpha - alpha;
                    similarity(t) = similarity(t) + (1+cos(delta))/2.0;
                    % segmentation
                    overlap_seg(t) = overlap_seg(t) + seg(det_idx, j);
                    overlap_box(t) = overlap_box(t) + seg_box(det_idx, j);
                    % 3D localization
                    t_det = det(det_idx, 7:9);
                    t_gt = gt(j).t;
                    error = norm(t_det - t_gt);
                    if error < 5
                        accuracy_5(t) = accuracy_5(t) + 1;
                    end
                    if error < 2
                        accuracy_2(t) = accuracy_2(t) + 1;
                    end
                    if error < 1
                        accuracy_1(t) = accuracy_1(t) + 1;
                    end
                end
            end
            
            % compute false positive
            for k = 1:num_det
                if assigned_detection(k) == 0 && det(k,6) >= thresholds(t)
                    fp(t) = fp(t) + 1;
                end
            end
            
            % do not consider detections overlapping with stuff area
            dontcare_gt = dontcare_gt_all{i};
            nstuff = 0;
            for j = 1:num
                if dontcare_gt(j) == 0
                    continue;
                end

                box_gt = [gt(j).x1 gt(j).y1 gt(j).x2 gt(j).y2];
                for k = 1:num_det
                    if assigned_detection(k) == 1
                        continue;
                    end
                    if det(k,6) < thresholds(t)
                        continue;
                    end
                    overlap = boxoverlap(det(k,:), box_gt);
                    if overlap > MIN_OVERLAP
                        assigned_detection(k) = 1;
                        nstuff = nstuff + 1;
                    end
                end
            end
            
            fp(t) = fp(t) - nstuff;
        end
    end
    
    for t = 1:nt
        % compute recall and precision
        recall(t) = tp(t) / (tp(t) + fn(t));
        precision(t) = tp(t) / (tp(t) + fp(t));
        fppi(t) = fp(t) / M;
        aos(t) = similarity(t) / (tp(t) + fp(t));
        asa(t) = overlap_seg(t) / (tp(t) + fp(t));
        asa_box(t) = overlap_box(t) / (tp(t) + fp(t));
        ala_5(t) = accuracy_5(t) / (tp(t) + fp(t));
        ala_2(t) = accuracy_2(t) / (tp(t) + fp(t));
        ala_1(t) = accuracy_1(t) / (tp(t) + fp(t));
    end
    
    % filter precision and aos
    for t = 1:nt
        precision(t) = max(precision(t:end));
        aos(t) = max(aos(t:end));
        asa(t) = max(asa(t:end));
        asa_box(t) = max(asa_box(t:end));
        ala_5(t) = max(ala_5(t:end));
        ala_2(t) = max(ala_2(t:end));
        ala_1(t) = max(ala_1(t:end));
    end
    
    recall_all{difficulty} = recall;
    precision_all{difficulty} = precision;
    fppi_all{difficulty} = fppi;
    aos_all{difficulty} = aos;
    asa_all{difficulty} = asa;
    asa_box_all{difficulty} = asa_box;
    ala_5_all{difficulty} = ala_5;
    ala_2_all{difficulty} = ala_2;
    ala_1_all{difficulty} = ala_1;
    thresholds_all{difficulty} = thresholds;
end

% average precision
recall_easy = recall_all{1};
recall_moderate = recall_all{2};
recall_hard = recall_all{3};
precision_easy = precision_all{1};
precision_moderate = precision_all{2};
precision_hard = precision_all{3};
ap_easy = VOCap(recall_easy, precision_easy);
fprintf('AP_easy = %.4f\n', ap_easy);
ap_moderate = VOCap(recall_moderate, precision_moderate);
fprintf('AP_moderate = %.4f\n', ap_moderate);
ap = VOCap(recall_hard, precision_hard);
fprintf('AP_hard = %.4f\n', ap);

fprintf('\n');

% average orientation similarity
aos_easy = aos_all{1};
aos_moderate = aos_all{2};
aos_hard = aos_all{3};
ap_easy = VOCap(recall_easy, aos_easy);
fprintf('AOS_easy = %.4f\n', ap_easy);
ap_moderate = VOCap(recall_moderate, aos_moderate);
fprintf('AOS_moderate = %.4f\n', ap_moderate);
ap = VOCap(recall_hard, aos_hard);
fprintf('AOS_hard = %.4f\n', ap);

fprintf('\n');


% average segmentation accuracy
asa_easy = asa_all{1};
asa_moderate = asa_all{2};
asa_hard = asa_all{3};
ap_easy = VOCap(recall_easy, asa_easy);
fprintf('ASA_easy = %.4f\n', ap_easy);
ap_moderate = VOCap(recall_moderate, asa_moderate);
fprintf('ASA_moderate = %.4f\n', ap_moderate);
ap = VOCap(recall_hard, asa_hard);
fprintf('ASA_hard = %.4f\n', ap);

fprintf('\n');

% average segmentation accuracy

asa_easy = asa_box_all{1};
asa_moderate = asa_box_all{2};
asa_hard = asa_box_all{3};
ap_easy = VOCap(recall_easy, asa_easy);
fprintf('ASA_box_easy = %.4f\n', ap_easy);
ap_moderate = VOCap(recall_moderate, asa_moderate);
fprintf('ASA_box_moderate = %.4f\n', ap_moderate);
ap = VOCap(recall_hard, asa_hard);
fprintf('ASA_box_hard = %.4f\n', ap);

fprintf('\n');

% average localization accuracy 5 meter

precision_easy = ala_5_all{1};
precision_moderate = ala_5_all{2};
precision_hard = ala_5_all{3};
ap_easy = VOCap(recall_easy, precision_easy);
fprintf('ALA_5_easy = %.4f\n', ap_easy);
ap_moderate = VOCap(recall_moderate, precision_moderate);
fprintf('ALA_5_moderate = %.4f\n', ap_moderate);
ap = VOCap(recall_hard, precision_hard);
fprintf('ALA_5_hard = %.4f\n', ap);

fprintf('\n');

% average localization accuracy 2 meter

precision_easy = ala_2_all{1};
precision_moderate = ala_2_all{2};
precision_hard = ala_2_all{3};
ap_easy = VOCap(recall_easy, precision_easy);
fprintf('ALA_2_easy = %.4f\n', ap_easy);
ap_moderate = VOCap(recall_moderate, precision_moderate);
fprintf('ALA_2_moderate = %.4f\n', ap_moderate);
ap = VOCap(recall_hard, precision_hard);
fprintf('ALA_2_hard = %.4f\n', ap);

fprintf('\n');


% average localization accuracy 1 meter

precision_easy = ala_1_all{1};
precision_moderate = ala_1_all{2};
precision_hard = ala_1_all{3};
ap_easy = VOCap(recall_easy, precision_easy);
fprintf('ALA_1_easy = %.4f\n', ap_easy);
ap_moderate = VOCap(recall_moderate, precision_moderate);
fprintf('ALA_1_moderate = %.4f\n', ap_moderate);
ap = VOCap(recall_hard, precision_hard);
fprintf('ALA_1_hard = %.4f\n', ap);

fprintf('\n');



function thresholds = get_thresholds(v, n_groundtruth, N_SAMPLE_PTS)

% sort scores in descending order
v = sort(v, 'descend');

% get scores for linearly spaced recall
current_recall = 0;
num = numel(v);
thresholds = [];
count = 0;
for i = 1:num

    % check if right-hand-side recall with respect to current recall is close than left-hand-side one
    % in this case, skip the current detection score
    l_recall = i / n_groundtruth;
    if i < num
      r_recall = (i+1) / n_groundtruth;
    else
      r_recall = l_recall;
    end

    if (r_recall - current_recall) < (current_recall - l_recall) && i < num
      continue;
    end

    % left recall is the best approximation, so use this and goto next recall step for approximation
    recall = l_recall;

    % the next recall step was reached
    count = count + 1;
    thresholds(count) = v(i);
    current_recall = current_recall + 1.0/(N_SAMPLE_PTS-1.0);
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