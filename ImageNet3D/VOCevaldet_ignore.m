function VOCevaldet_ignore(network, region_proposal, minoverlap)

matlabpool open;

opt = globals();
root = opt.path_imagenet3d;

result_dir = '/scail/scratch/u/yuxiang/3DVP_RCNN/fast-rcnn/output/imagenet3d/imagenet3d_test';
if exist(result_dir, 'dir') == 0
    result_dir = '/home/yuxiang/Projects/3DVP_RCNN/fast-rcnn/output/imagenet3d/imagenet3d_test';
end
method = sprintf('%s_fast_rcnn_original_imagenet3d_%s_iter_160000', network, region_proposal);

% load class name
classes = textread(sprintf('%s/Image_sets/classes.txt', root), '%s');
num_cls = numel(classes);

% load test set
gtids = textread(sprintf('%s/Image_sets/test.txt', root), '%s');
M = numel(gtids);

% read ground truth
recs = cell(1, M);
count = 0;
for i = 1:M
    % read ground truth 
    filename = sprintf('%s/Annotations/%s.mat', root, gtids{i});
    object = load(filename);
    recs{i} = object.record;
    count = count + numel(object.record.objects);
end
fprintf('load ground truth done, %d objects\n', count);

recalls = cell(num_cls, 1);
precisions = cell(num_cls, 1);
aps = zeros(num_cls, 1);
parfor k = 1:num_cls
    cls = classes{k};
    
    % extract ground truth objects
    npos = 0;
    npos_view = 0;
    gt = [];
    for i = 1:M
        % extract objects of class
        clsinds = strmatch(cls, {recs{i}.objects(:).class}, 'exact');
        gt(i).BB = cat(1, recs{i}.objects(clsinds).bbox)';
        gt(i).det = false(length(clsinds), 1);
        gt(i).ignore = false(length(clsinds), 1);
        
        % viewpoint
        num = length(clsinds);
        for j = 1:num
            viewpoint = recs{i}.objects(j).viewpoint;
            if isempty(viewpoint) == 1
                gt(i).ignore(j) = true;
                continue;
            end
            npos_view = npos_view + 1;
        end
        
        npos = npos + length(clsinds);
    end

    % load detections
    filename = sprintf('%s/%s/detections_%s.txt', result_dir, method, cls);
    fid = fopen(filename, 'r');
    C = textscan(fid, '%s %f %f %f %f %f');
    fclose(fid);
    
    ids = C{1};
    b1 = C{2};
    b2 = C{3};
    b3 = C{4};
    b4 = C{5};
    confidence = C{6};
    BB = [b1 b2 b3 b4]';

    % sort detections by decreasing confidence
    [~, si]=sort(-confidence);
    ids = ids(si);
    BB = BB(:,si);

    % assign detections to ground truth objects
    nd = length(confidence);
    tp = zeros(nd, 1);
    fp = zeros(nd, 1);
    ignore = false(nd, 1);
    tic;
    for d = 1:nd
        % display progress
        if toc > 1
            fprintf('%s: pr: compute: %d/%d\n', cls, d, nd);
            tic;
        end

        % find ground truth image
        i = find(strcmp(ids{d}, gtids) == 1);
        if isempty(i)
            error('unrecognized image "%s"', ids{d});
        elseif length(i)>1
            error('multiple image "%s"', ids{d});
        end

        % assign detection to ground truth object if any
        bb = BB(:,d);
        ovmax = -inf;
        jmax = -1;
        for j = 1:size(gt(i).BB, 2)
            bbgt = gt(i).BB(:,j);
            bi = [max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
            iw = bi(3) - bi(1) + 1;
            ih = bi(4) - bi(2) + 1;
            if iw > 0 && ih > 0                
                % compute overlap as area of intersection / area of union
                ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
                   (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
                   iw*ih;
                ov= iw * ih / ua;
                if ov > ovmax
                    ovmax = ov;
                    jmax = j;
                end
            end
        end
        % assign detection as true positive/don't care/false positive
        if ovmax >= minoverlap
            if ~gt(i).det(jmax)
                tp(d) = 1;            % true positive
                gt(i).det(jmax) = true;
            else
                fp(d) = 1;            % false positive (multiple detection)
            end
            if gt(i).ignore(jmax)
                ignore(d) = true;
            end            
        else
            fp(d) = 1;                % false positive
        end
    end

    % compute precision/recall
    fp = cumsum(fp(~ignore));
    tp = cumsum(tp(~ignore));
    rec = tp / npos_view;
    prec = tp ./ (fp + tp);
    aps(k) = VOCap(rec, prec);
    recalls{k} = rec;
    precisions{k} = prec;
    fprintf('%s, ap: %f\n', cls, aps(k));
end

% write to file
fid = fopen(sprintf('aps_%s_%d_ignore.txt', method, minoverlap*100), 'w');
for i = 1:num_cls
    fprintf(fid, '%s %f\n', classes{i}, aps(i));
end
fprintf(fid, 'mAP %f\n', mean(aps));
fclose(fid);

% save to matfile
matfile = sprintf('aps_%s_%d_ignore.mat', method, minoverlap*100);
save(matfile, 'recalls', 'precisions', 'aps', '-v7.3');

matlabpool close;
