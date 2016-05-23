% compute recall and precision
function compute_recall_precision

opt = globals();
is_train = 1;
is_show = 0;

if is_train
    seq_set = 'train';
    N = numel(opt.mot2d_train_seqs);
else
    seq_set = 'test';
    N = numel(opt.mot2d_test_seqs);
end

% output dir
out_dir = 'detection_train';

% main loop
for seq_idx = 1:N
    
    if is_train
        seq_name = opt.mot2d_train_seqs{seq_idx};
        seq_num = opt.mot2d_train_nums(seq_idx);
    else
        seq_name = opt.mot2d_test_seqs{seq_idx};
        seq_num = opt.mot2d_test_nums(seq_idx);
    end

    % load detection results
    filename = fullfile(out_dir, seq_set, [seq_name '.txt']);
    [frame_id, ~, b1, b2, b3, b4, confidence, ~, ~, ~] = textread(filename, '%d %d %f %f %f %f %f %f %f %f');
    % filename = fullfile(opt.mot, opt.mot2d, seq_set, seq_name, 'det', 'det.txt');
    % [frame_id, ~, b1, b2, b3, b4, confidence, ~, ~, ~] = textread(filename, '%d %d %f %f %f %f %f %f %f %f', 'delimiter', ',');
    b3 = b3 + b1;
    b4 = b4 + b2;
    
    % read ground truth
    filename = fullfile(opt.mot, opt.mot2d, seq_set, seq_name, 'gt', 'gt.txt');
    fid = fopen(filename, 'r');
    Cgt = textscan(fid, '%d %d %f %f %f %f %f %f %f %f', 'delimiter', ',');
    fclose(fid);
    % fprintf('load ground truth done\n');    

    energy = [];
    correct = [];
    overlap = [];
    M = seq_num;
    count = zeros(M,1);
    num = zeros(M,1);
    num_pr = 0;
    for i = 1:M
        % fprintf('%s: %d/%d\n', seq_name, i, M);    
        % read ground truth bounding box        
        index = find(Cgt{1} == i);
        bbox = [Cgt{3}(index) Cgt{4}(index) Cgt{3}(index)+Cgt{5}(index) Cgt{4}(index)+Cgt{6}(index)];
        count(i) = size(bbox, 1);
        det = zeros(count(i), 1);

        % get predicted bounding box
        index = frame_id == i;
        dets = [b1(index) b2(index) b3(index) b4(index) confidence(index)];
        num(i) = size(dets, 1);

        % for each predicted bounding box
        for j = 1:num(i)
            num_pr = num_pr + 1;
            energy(num_pr) = dets(j, 5);
            bbox_pr = dets(j, 1:4);

            % compute box overlap
            if isempty(bbox) == 0
                o = boxoverlap(bbox, bbox_pr);
                [maxo, index] = max(o);
                if maxo >= 0.5 && det(index) == 0
                    overlap{num_pr} = index;
                    correct(num_pr) = 1;
                    det(index) = 1;
                else
                    overlap{num_pr} = [];
                    correct(num_pr) = 0;
                end
            else
                overlap{num_pr} = [];
                correct(num_pr) = 0;
            end
        end
    end
    overlap = overlap';

    [threshold, index] = sort(energy, 'descend');
    correct = correct(index);
    n = numel(threshold);
    recall = zeros(n,1);
    precision = zeros(n,1);
    num_correct = 0;
    for i = 1:n
        % compute precision
        num_positive = i;
        num_correct = num_correct + correct(i);
        if num_positive ~= 0
            precision(i) = num_correct / num_positive;
        else
            precision(i) = 0;
        end

        % compute recall
        recall(i) = num_correct / sum(count);
    end


    ap = VOCap(recall, precision);
    fprintf('%s %.4f\n', seq_name, ap);

    % draw recall-precision and accuracy curve
    if is_show
        figure(1);
        hold on;
        plot(recall, precision, 'r', 'LineWidth',3);
        xlabel('Recall');
        ylabel('Precision');
        tit = sprintf('Average Precision = %.1f', 100*ap);
        title(tit);
        hold off;
        pause;
    end
end