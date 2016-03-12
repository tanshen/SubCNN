function compute_recall_precision_aos_box

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
object = load(fullfile(SLMroot, 'KITTI/data.mat'));
data = object.data;
idx = data.idx_ap;
centers = unique(idx);
centers(centers == -1) = [];

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
result_dir = 'results_faster_rcnn';
detections = cell(1, M);
for i = 1:M
    filename = sprintf('%s/%06d.txt', result_dir, ids(i));
    disp(filename);
    fid = fopen(filename, 'r');
    C = textscan(fid, '%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f');   
    fclose(fid);
    
    det = double([C{5} C{6} C{7} C{8} ones(numel(C{1}),1) C{end}]);
    index = strcmp('Car', C{1});
    det = det(index, :);
    detections{i} = det;
end
fprintf('load detection done\n');

% read segmentation scores
segmentations_box = cell(1, M);
for i = 1:M
    filename = sprintf('%s/%06d_seg.mat', result_dir, i);
    object = load(filename);
    segmentations_box{i} = object.Scores_box;
end
fprintf('load segmentation scores done\n');

recall_all = cell(1, 3);
precision_all = cell(1, 3);
aos_all = cell(1, 3);
asa_box_all = cell(1, 3);

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

        % get predicted bounding box
        det = detections{i};
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
    overlap_box = zeros(nt, 1);
    recall = zeros(nt, 1);
    precision = zeros(nt, 1);
    aos = zeros(nt, 1);
    asa_box = zeros(nt, 1);
    
    % for each image
    for i = 1:M
        disp(i);
        gt = groundtruths{i};
        num = numel(gt);
        ignored_gt = ignored_gt_all{i};
        
        det = detections{i};
        det = truncate_detections(det);    
        num_det = size(det, 1);
        
        seg_box = segmentations_box{i};
        if num && num_det
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
                    cid = det(det_idx, 5);
                    azimuth = data.azimuth(centers(cid));
                    alpha = azimuth + 90;
                    if alpha >= 360
                        alpha = alpha - 360;
                    end
                    alpha = alpha*pi/180;
                    if alpha > pi
                        alpha = alpha - 2*pi;
                    end
                    delta = gt(j).alpha - alpha;
                    similarity(t) = similarity(t) + (1+cos(delta))/2.0;
                    % segmentation
                    overlap_box(t) = overlap_box(t) + seg_box(det_idx, j);
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
        aos(t) = similarity(t) / (tp(t) + fp(t));
        asa_box(t) = overlap_box(t) / (tp(t) + fp(t));
    end
    
    % filter precision and aos
    for t = 1:nt
        precision(t) = max(precision(t:end));
        aos(t) = max(aos(t:end));
        asa_box(t) = max(asa_box(t:end));
    end
    
    recall_all{difficulty} = recall;
    precision_all{difficulty} = precision;
    aos_all{difficulty} = aos;
    asa_box_all{difficulty} = asa_box;
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

figure(1);
hold on;
plot(recall_easy, precision_easy, 'g', 'LineWidth',3);
plot(recall_moderate, precision_moderate, 'b', 'LineWidth',3);
plot(recall_hard, precision_hard, 'r', 'LineWidth',3);
h = xlabel('Recall');
set(h, 'FontSize', 12);
h = ylabel('Precision');
set(h, 'FontSize', 12);
tit = sprintf('Car APs');
h = title(tit);
set(h, 'FontSize', 12);
hold off;

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

% draw recall-precision and accuracy curve
figure(2);
hold on;
plot(recall_easy, aos_easy, 'g', 'LineWidth',3);
plot(recall_moderate, aos_moderate, 'b', 'LineWidth',3);
plot(recall_hard, aos_hard, 'r', 'LineWidth',3);
h = xlabel('Recall');
set(h, 'FontSize', 12);
h = ylabel('AOS');
set(h, 'FontSize', 12);
tit = sprintf('Car AOSs');
h = title(tit);
set(h, 'FontSize', 12);
hold off;

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

figure(3);
hold on;
plot(recall_easy, asa_easy, 'g', 'LineWidth',3);
plot(recall_moderate, asa_moderate, 'b', 'LineWidth',3);
plot(recall_hard, asa_hard, 'r', 'LineWidth',3);
h = xlabel('Recall');
set(h, 'FontSize', 12);
h = ylabel('ASA Box');
set(h, 'FontSize', 12);
tit = sprintf('Car ASA Box');
h = title(tit);
set(h, 'FontSize', 12);
hold off;


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