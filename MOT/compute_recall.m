function compute_recall

opt = globals;
seq_set = 'train';
out_dir = 'region_proposals/SubCNN';
N = numel(opt.mot2d_train_seqs);

for seq_idx = 1:N
    
    seq_name = opt.mot2d_train_seqs{seq_idx};
    seq_num = opt.mot2d_train_nums(seq_idx);
    disp(seq_name);

    % read ground truth
    filename = fullfile(opt.mot, opt.mot2d, seq_set, seq_name, 'gt', 'gt.txt');
    fid = fopen(filename, 'r');
    Cgt = textscan(fid, '%d %d %f %f %f %f %f %f %f %f', 'delimiter', ',');
    fclose(fid);
    fprintf('load ground truth done\n');

    % read region proposals
    filename = fullfile(out_dir, seq_set, [seq_name '.txt']);
    disp(filename);
    fid = fopen(filename, 'r');
    Cdet = textscan(fid, '%d %d %f %f %f %f %f %f %f %f', 'delimiter', ',');
    fclose(fid);   
    fprintf('load region proposals done\n');

    % for each image
    num_boxes_all = 0;
    num_boxes_covered = 0;
    for i = 1:seq_num
        % collect ground truth boxes
        index = Cgt{1} == i;
        box_gt = [Cgt{3}(index) Cgt{4}(index) Cgt{3}(index)+Cgt{5}(index) Cgt{4}(index)+Cgt{6}(index)];

        % detections
        index = Cdet{1} == i;
        det = [Cdet{3}(index) Cdet{4}(index) Cdet{3}(index)+Cdet{5}(index) Cdet{4}(index)+Cdet{6}(index)];
        num_det = size(det, 1);

        % compute statistics
        % for each ground truth
        for j = 1:size(box_gt, 1)
            num_boxes_all = num_boxes_all + 1;        

            if num_det == 0
                overlap = 0;
            else
                overlap = boxoverlap(det, box_gt(j,:));
            end
            % disp(max(overlap));
            if max(overlap) >= 0.5
                num_boxes_covered = num_boxes_covered + 1;
            end
        end
    end

    % compute recall
    recall = num_boxes_covered / num_boxes_all;
    fprintf('%s: %f\n', seq_name, recall);
end