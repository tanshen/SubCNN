function exemplar_display_result_nissan

threshold = 0.1;
is_save = 1;
result_dir = 'test_results';
root_dir = '/home/yuxiang/Projects/NISSAN_Dataset/Images';
image_set = '2015-10-21-16-25-12';

% read detection results
filename = sprintf('%s/detections.txt', result_dir);
[ids_det, cls_det, x1_det, y1_det, x2_det, y2_det, cid_det, score_det] = ...
    textread(filename, '%s %s %f %f %f %f %d %f');
fprintf('load detection done\n');

% read ids
filename = sprintf('%s/%s.txt', root_dir, image_set);
fid = fopen(filename, 'r');
C = textscan(fid, '%s');
fclose(fid);
ids = C{1};
N = numel(ids);

% load data
exemplar_globals;
filename = fullfile(SLMroot, 'KITTI/data_kitti.mat');
object = load(filename);
data = object.data;
centers = unique(data.idx_ap);
centers(centers == -1) = [];
fprintf('%d clusters\n', numel(centers));

hf = figure(1);
cmap = colormap(summer);
for i = 1:N
    img_idx = ids{i};
    disp(img_idx);
    
    % get predicted bounding box
    index = strcmp(img_idx, ids_det);
    det_cls = cls_det(index);
    det = [x1_det(index), y1_det(index), x2_det(index), y2_det(index), cid_det(index), score_det(index)];
    if isempty(det) == 1
        fprintf('no detection for image %s\n', img_idx);
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
    
    file_img = sprintf('%s/%s/%s.bmp', root_dir, image_set, img_idx);
    I = imread(file_img);

    % add pattern
    for k = 1:num
        if det(k,6) > threshold
            bbox_pr = det(k,1:4);
            bbox = zeros(1,4);
            bbox(1) = max(1, floor(bbox_pr(1)));
            bbox(2) = max(1, floor(bbox_pr(2)));
            bbox(3) = min(size(I,2), floor(bbox_pr(3)));
            bbox(4) = min(size(I,1), floor(bbox_pr(4)));
            w = bbox(3) - bbox(1) + 1;
            h = bbox(4) - bbox(2) + 1;
            
            if strcmp(det_cls{k}, 'Car') == 0
                continue;
            end

            % apply the 2D occlusion mask to the bounding box
            % check if truncated pattern
            cid = centers(det(k,5));
            pattern = data.pattern{cid};                
            index = find(pattern == 1);
            if data.truncation(cid) > 0 && isempty(index) == 0
                
                [y, x] = ind2sub(size(pattern), index);
                cx = size(pattern, 2)/2;
                cy = size(pattern, 1)/2;
                width = size(pattern, 2);
                height = size(pattern, 1);                 
                pattern = pattern(min(y):max(y), min(x):max(x));

                % find the object center
                sx = w / size(pattern, 2);
                sy = h / size(pattern, 1);
                tx = bbox(1) - sx*min(x);
                ty = bbox(2) - sy*min(y);
                cx = sx * cx + tx;
                cy = sy * cy + ty;
                width = sx * width;
                height = sy * height;
                bbox_pr = round([cx-width/2 cy-height/2 cx+width/2 cy+height/2]);
                width = bbox_pr(3) - bbox_pr(1) + 1;
                height = bbox_pr(4) - bbox_pr(2) + 1;
                
                pattern = imresize(data.pattern{cid}, [height width], 'nearest');
                
                bbox = zeros(1,4);
                bbox(1) = max(1, floor(bbox_pr(1)));
                start_x = bbox(1) - floor(bbox_pr(1)) + 1;
                bbox(2) = max(1, floor(bbox_pr(2)));
                start_y = bbox(2) - floor(bbox_pr(2)) + 1;
                bbox(3) = min(size(I,2), floor(bbox_pr(3)));
                bbox(4) = min(size(I,1), floor(bbox_pr(4)));
                w = bbox(3) - bbox(1) + 1;
                h = bbox(4) - bbox(2) + 1;
                pattern = pattern(start_y:start_y+h-1, start_x:start_x+w-1);
            else
                pattern = imresize(pattern, [h w], 'nearest');
            end
            
            % build the pattern in the image
            height = size(I,1);
            width = size(I,2);
            P = uint8(zeros(height, width));
            x = bbox(1);
            y = bbox(2);
            index_y = y:min(y+h-1, height);
            index_x = x:min(x+w-1, width);
            P(index_y, index_x) = pattern(1:numel(index_y), 1:numel(index_x));
            
            % show occluded region
            im = create_occlusion_image(pattern);
            x = bbox(1);
            y = bbox(2);
            Isub = I(y:y+h-1, x:x+w-1, :);
            index = im == 255;
            im(index) = Isub(index);
            I(y:y+h-1, x:x+w-1, :) = uint8(0.1*Isub + 0.9*im);             
            
            % show segments
            index_color = 1 + floor((k-1) * size(cmap,1) / num);
            dispColor = 255*cmap(index_color,:);
            scale = round(max(size(I))/400);            
            [gx, gy] = gradient(double(P));
            g = gx.^2 + gy.^2;
            g = conv2(g, ones(scale), 'same');
            edgepix = find(g > 0);
            npix = numel(P);
            for b = 1:3
                I((b-1)*npix+edgepix) = dispColor(b);
            end
        end
    end
    
    imshow(I);
    hold on;
    for k = 1:num
        if det(k,6) > threshold && strcmp(det_cls{k}, 'Car') == 0
            % get predicted bounding box
            bbox_pr = det(k,1:4);
            bbox_draw = [bbox_pr(1), bbox_pr(2), bbox_pr(3)-bbox_pr(1), bbox_pr(4)-bbox_pr(2)];
            if strcmp(det_cls{k}, 'Pedestrian') == 1
                rectangle('Position', bbox_draw, 'EdgeColor', 'm', 'LineWidth', 4);
            else
                rectangle('Position', bbox_draw, 'EdgeColor', [.7 .5 0], 'LineWidth', 4);
            end
        end
    end
    hold off;
    
    if is_save
        filename = fullfile('result_images', sprintf('%s.png', img_idx));
        % saveas(hf, filename);
        hgexport(hf, filename, hgexport('factorystyle'), 'Format', 'png');
    else
        pause;
    end
end

function im = create_occlusion_image(pattern)

% 2D occlusion mask
im = 255*ones(size(pattern,1), size(pattern,2), 3);
color = [0 0 255];
for j = 1:3
    tmp = im(:,:,j);
    tmp(pattern == 2) = color(j);
    im(:,:,j) = tmp;
end
color = [0 255 255];
for j = 1:3
    tmp = im(:,:,j);
    tmp(pattern == 3) = color(j);
    im(:,:,j) = tmp;
end
im = uint8(im);  