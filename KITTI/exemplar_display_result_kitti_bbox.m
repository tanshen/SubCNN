function exemplar_display_result_kitti_bbox

threshold = 0.5;
is_save = 0;

% read detection results
result_dir = 'test_results_5';
filename = sprintf('%s/dets_3d.mat', result_dir);
object = load(filename);
dets = object.dets_3d;
fprintf('load detection done\n');

% read ids of validation images
object = load('kitti_ids_new.mat');
ids = object.ids_test;
N = numel(ids);

% KITTI path
exemplar_globals;
root_dir = KITTIroot;
data_set = 'testing';
cam = 2;
image_dir = fullfile(root_dir, [data_set '/image_' num2str(cam)]);

figure;
cmap = colormap(summer);
for i = 6405 %[1692, 6299, 6405] %2739:N
    img_idx = ids(i);
    disp(img_idx);
    
    % get predicted bounding box
    objects = dets{i};
    num = numel(objects);    
    det = zeros(num, 6);
    det_cls = cell(num, 1);
    for k = 1:num
        det(k,:) = [objects(k).x1 objects(k).y1 objects(k).x2 objects(k).y2 ...
                objects(k).cid objects(k).score];
        det_cls{k} = objects(k).type;
    end    
    
    if isempty(det) == 1
        fprintf('no detection for image %d\n', img_idx);
        continue;
    end
    if max(det(:,6)) < threshold
        fprintf('maximum score %.2f is smaller than threshold\n', max(det(:,6)));
        continue;
    end
    if isempty(det) == 0
        I = det(:,6) >= threshold;
        det = det(I,:);
        det_cls = det_cls(I);
        height = det(:,4) - det(:,2);
        [~, I] = sort(height);
        det = det(I,:);
        det_cls = det_cls(I);
    end
    num = size(det, 1);
    
    file_img = sprintf('%s/%06d.png', image_dir, img_idx);
    I = imread(file_img);
    
    % show all the detections
%     figure(1);
%     imshow(I);
%     hold on;
%     
%     for k = 1:size(dets{i},1)
%         bbox_pr = dets{i}(k,1:4);
%         bbox_draw = [bbox_pr(1), bbox_pr(2), bbox_pr(3)-bbox_pr(1), bbox_pr(4)-bbox_pr(2)];
%         rectangle('Position', bbox_draw, 'EdgeColor', 'g', 'LineWidth', 2);
%     end
%     hold off;
    
    imshow(I);
    hold on;
    for k = 1:num
        if det(k,6) > threshold
            % get predicted bounding box
            bbox_pr = det(k,1:4);
            bbox_pr(1) = max(1, bbox_pr(1));
            bbox_pr(2) = max(1, bbox_pr(2));
            bbox_pr(3) = min(size(I,2), bbox_pr(3));
            bbox_pr(4) = min(size(I,1), bbox_pr(4));
            bbox_draw = [bbox_pr(1), bbox_pr(2), bbox_pr(3)-bbox_pr(1), bbox_pr(4)-bbox_pr(2)];

            if strcmp(det_cls{k}, 'Car') == 1
                index_color = 1 + floor((k-1) * size(cmap,1) / num);
                rectangle('Position', bbox_draw, 'EdgeColor', cmap(index_color,:), 'LineWidth', 6);
            elseif strcmp(det_cls{k}, 'Pedestrian') == 1
                rectangle('Position', bbox_draw, 'EdgeColor', 'm', 'LineWidth', 6);
            elseif strcmp(det_cls{k}, 'Cyclist') == 1
                rectangle('Position', bbox_draw, 'EdgeColor', [.7 .5 0], 'LineWidth', 6);
            end
%             s = sprintf('%.2f', det(k,6));
%             text(bbox_pr(1), bbox_pr(2), s, 'FontSize', 4, 'BackgroundColor', 'c');
        end
    end
    
    hold off;
    
    if is_save
        filename = fullfile('result_images', sprintf('%06d.png', img_idx));
        saveas(hf, filename);
    else
        pause;
    end
end  