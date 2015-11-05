function compute_ap_avp(vnum_train)

vnum_test = vnum_train;

classes = {'aeroplane', 'bicycle', 'boat', ...
           'bottle', 'bus', 'car', 'chair', ...
           'diningtable', 'motorbike', ...
           'sofa', 'train', 'tvmonitor'};
num_cls = numel(classes);

aps = zeros(num_cls, 1);
avps = zeros(num_cls, 1);
for i = 1:num_cls
    cls = classes{i};
    [recall, precision, accuracy, ap, aa] = compute_recall_precision_accuracy(cls, vnum_train, vnum_test);
    aps(i) = ap;
    avps(i) = aa;
end

for i = 1:num_cls
    cls = classes{i};
    fprintf('%s %f %f\n', cls, aps(i), avps(i));
end