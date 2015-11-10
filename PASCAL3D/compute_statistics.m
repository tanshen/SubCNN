function compute_statistics

opt = globals;
pascal_init;

classes = {'aeroplane', 'bicycle', 'boat', ...
           'bottle', 'bus', 'car', 'chair', ...
           'diningtable', 'motorbike', ...
           'sofa', 'train', 'tvmonitor'};
num_cls = numel(classes);
nums = zeros(num_cls, 1);

% load test set
[ids, t] = textread(sprintf(VOCopts.imgsetpath, 'train'), '%s %d');
M = numel(ids);

% read ground truth
for i = 1:M
    % read ground truth 
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    objects = rec.objects;
    
    for j = 1:numel(objects)
        difficult = objects(j).difficult;
        cls = objects(j).class;
        index = find(strcmp(cls, classes) == 1);
        if isempty(index) == 0
            nums(index) = nums(index) + 1;
        end
    end    
end

fprintf('%d images\n', M);
for i = 1:numel(classes)
    fprintf('%s: %d objects\n', classes{i}, nums(i));
end