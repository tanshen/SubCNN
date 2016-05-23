function show_detections

opt = globals();
is_train = 1;
threshold = 0.5;
K = 200;

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
    
    % read results
    filename = fullfile(out_dir, seq_set, [seq_name '.txt']);
    disp(filename);
    fid = fopen(filename, 'r');
    C = textscan(fid, '%d %d %f %f %f %f %f %f %f %f');
    fclose(fid);  
    
    for i = 1:seq_num
        filename = fullfile(opt.mot, opt.mot2d, seq_set, seq_name, 'img1', sprintf('%06d.jpg', i));
        disp(filename);
        I = imread(filename);
        imshow(I);
        hold on;
        
        index = C{1} == i;
        boxes = [C{3}(index) C{4}(index) C{5}(index) C{6}(index)];
        scores = C{7}(index);
        for k = 1:min(K, size(boxes, 1))
            box = boxes(k, :);
            if box(3) > 0 && box(4) > 0 && scores(k) > threshold
                rectangle('Position', box, 'EdgeColor', 'g', 'LineWidth', 2);
            end
        end
        
        pause;
    end
        
end