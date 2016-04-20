function selective_search_MOT

matlabpool open;

opt = globals();
is_train = 1;

if is_train
    seq_set = 'train';
    N = numel(opt.mot2d_train_seqs);
else
    seq_set = 'test';
    N = numel(opt.mot2d_test_seqs);
end

% output dir
out_dir = 'region_proposals/selective_search';
if exist(out_dir, 'dir') == 0
    mkdir(out_dir);
end

% main loop
for seq_idx = 1:N
    
    if is_train
        seq_name = opt.mot2d_train_seqs{seq_idx};
        seq_num = opt.mot2d_train_nums(seq_idx);
    else
        seq_name = opt.mot2d_test_seqs{seq_idx};
        seq_num = opt.mot2d_test_nums(seq_idx);
    end
    
    boxes = cell(1, seq_num);
    parfor i = 1:seq_num
        filename = fullfile(opt.mot, opt.mot2d, seq_set, seq_name, 'img1', sprintf('%06d.jpg', i));
        disp(filename);
        I = imread(filename);
        boxes{i} = selective_search_boxes(I);
        fprintf('%s: %d \\ %d, %d boxes\n', seq_name, i, seq_num, size(boxes{i}, 1));
    end
    
    if exist(fullfile(out_dir, seq_set), 'dir') == 0
        mkdir(fullfile(out_dir, seq_set));
    end    
    
    % write results
    filename = fullfile(out_dir, seq_set, [seq_name '.txt']);
    fid = fopen(filename, 'w');
    for i = 1:seq_num
        box = boxes{i};
        for j = 1:size(box,1)
            fprintf(fid, '%d, -1, %.2f, %.2f, %.2f, %.2f, -1, -1, -1, -1\n', ...
                i, box(j,2), box(j,1), box(j,4)-box(j,2), box(j,3)-box(j,1));
        end
    end
    fclose(fid);
end

matlabpool close;