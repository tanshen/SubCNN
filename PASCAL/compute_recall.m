function compute_recall

opt = globals;
pascal_init;

classes = {'aeroplane', 'bicycle', 'bird', 'boat', ...
           'bottle', 'bus', 'car', 'cat', 'chair', ...
           'cow', 'diningtable', 'dog', 'horse', ...
           'motorbike', 'person', 'pottedplant', ...
           'sheep', 'sofa', 'train', 'tvmonitor'};
num_cls = numel(classes);

% load test set
[gtids, t] = textread(sprintf(VOCopts.imgsetpath, 'test'), '%s %d');
M = numel(gtids);

% read ground truth
groundtruths = cell(1, M);
for i = 1:M
    % read ground truth 
    rec = PASreadrecord(sprintf(VOCopts.annopath, gtids{i}));
    groundtruths{i} = rec.objects;
end
fprintf('load ground truth done\n');

% read region proposals
count = 0;
detections = cell(1, M);
min_score = inf;
for i = 1:M
    filename = sprintf('region_proposals/%s.txt', gtids{i});
    disp(filename);
    fid = fopen(filename, 'r');
    C = textscan(fid, '%f %f %f %f %f');   
    fclose(fid);
    
    det = double([C{1} C{2} C{3} C{4} C{5}]);
    min_score = min(min_score, min(C{5}));
    ind = (det(:,3) > det(:,1)) & (det(:,4) > det(:,2));
    det = det(ind,1:4);

    detections{i} = det;
    count = count + size(detections{i}, 1);
end
fprintf('load detection done\n');
fprintf('%f detections per image, with min score %f\n', count / M, min_score);

% for each image
num_boxes_all = zeros(num_cls, 1);
num_boxes_covered = zeros(num_cls, 1);
for i = 1:M
    % collect ground truth boxes
    gt = groundtruths{i};
    num_gt = numel(gt);
    box_gt = zeros(num_gt, 4);
    difficult_gt = zeros(num_gt, 1);
    cls_gt = cell(num_gt, 1);
    for j = 1:num_gt
        box_gt(j,:) = gt(j).bbox;
        difficult_gt(j) = gt(j).difficult;
        cls_gt{j} = gt(j).class;
    end
    
    index = find(difficult_gt == 0);
    box_gt = box_gt(index, :);
    cls_gt = cls_gt(index);

    % detections
    det = detections{i};
    num_det = size(det, 1);

    % compute statistics
    % for each ground truth
    for j = 1:numel(cls_gt)
        cls = cls_gt{j};
        index = strcmp(cls, classes) == 1;
        num_boxes_all(index) = num_boxes_all(index) + 1;        
        
        if num_det == 0
            overlap = 0;
        else
            overlap = boxoverlap(det, box_gt(j,:));
        end
        % disp(max(overlap));
        if max(overlap) >= VOCopts.minoverlap
            num_boxes_covered(index) = num_boxes_covered(index) + 1;
        end
    end
end

% compute recall
for i = 1:num_cls
    recall = num_boxes_covered(i) / num_boxes_all(i);
    fprintf('%s: %f\n', classes{i}, recall);
end