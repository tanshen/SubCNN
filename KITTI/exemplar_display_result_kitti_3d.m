function exemplar_display_result_kitti_3d

is_save = 0;

threshold = 0.5;
cls = 'car';

% read detection results
result_dir = 'test_results_5';
filename = sprintf('%s/dets_3d.mat', result_dir);
object = load(filename);
dets_all = object.dets_3d;
fprintf('load detection done\n');

% read ids of validation images
object = load('kitti_ids_new.mat');
ids = object.ids_test;
N = numel(ids);

% load PASCAL3D+ cad models
exemplar_globals;
filename = sprintf(path_cad, cls);
object = load(filename);
cads = object.(cls);
cads([7, 8, 10]) = [];

filename = '../CAD/pedestrian.off';
[vertices_pedestrian, faces_pedestrian] = load_off_file(filename);

filename = '../CAD/cyclist.off';
[vertices_cyclist, faces_cyclist] = load_off_file(filename);

% KITTI path
root_dir = KITTIroot;
data_set = 'testing';
cam = 2;
image_dir = fullfile(root_dir, [data_set '/image_' num2str(cam)]);
calib_dir = fullfile(root_dir,[data_set '/calib']);

% load data
filename = fullfile(SLMroot, 'KITTI/data_kitti.mat');
object = load(filename);
data = object.data;

filename = fullfile(SLMroot, 'KITTI/data_kitti_pedestrian.mat');
object = load(filename);
data_pedestrian = object.data;

filename = fullfile(SLMroot, 'KITTI/data_kitti_cyclist.mat');
object = load(filename);
data_cyclist = object.data;

hf = figure;
cmap = colormap(summer);
ind_plot = 1;
mplot = 2;
nplot = 2;

for i = [1692, 6299, 6405] %1:N
    disp(i);
    img_idx = ids(i);
    
    objects = dets_all{i};
    num = numel(objects);
    
    % construct detections
    dets = zeros(num, 6);
    det_cls = cell(num, 1);
    for k = 1:num
        if isempty(objects(k).score) == 1
            continue;
        end
        dets(k,:) = [objects(k).x1 objects(k).y1 objects(k).x2 objects(k).y2 ...
                objects(k).cid objects(k).score];
        det_cls{k} = objects(k).type;
    end
    
    if max(dets(:,6)) < threshold
        fprintf('maximum score %.2f is smaller than threshold\n', max(dets(:,6)));
        continue;
    end
    
    if isempty(dets) == 0
        I = dets(:,6) >= threshold;
        dets = dets(I,:);
        det_cls = det_cls(I);
        height = dets(:,4) - dets(:,2);
        [~, I] = sort(height);
        dets = dets(I,:);
        det_cls = det_cls(I);
    end
    num = size(dets, 1);
    
    % load the velo_to_cam matrix
    R0_rect = readCalibration(calib_dir, img_idx, 4);
    tmp = R0_rect';
    tmp = tmp(1:9);
    tmp = reshape(tmp, 3, 3);
    tmp = tmp';
    Pv2c = readCalibration(calib_dir, img_idx, 5);
    Pv2c = tmp * Pv2c;
    Pv2c = [Pv2c; 0 0 0 1];
    
    % camera location in world
    C = Pv2c\[0; 0; 0; 1];
    C(4) = [];    
    
    % plot 2D detections
    subplot(mplot, nplot, [1 2]);
    file_img = sprintf('%s/%06d.png', image_dir, img_idx);
    I = imread(file_img);
    for k = 1:num
        if dets(k,6) > threshold
            bbox_pr = dets(k,1:4);
            bbox = zeros(1,4);
            bbox(1) = max(1, floor(bbox_pr(1)));
            bbox(2) = max(1, floor(bbox_pr(2)));
            bbox(3) = min(size(I,2), floor(bbox_pr(3)));
            bbox(4) = min(size(I,1), floor(bbox_pr(4)));
            w = bbox(3) - bbox(1) + 1;
            h = bbox(4) - bbox(2) + 1;
            
            if strcmp(det_cls{k}, 'Car') == 0 % && strcmp(det_cls{k}, 'Cyclist') == 0
                continue;
            end            

            % apply the 2D occlusion mask to the bounding box
            % check if truncated pattern
            cid = dets(k,5);
            if strcmp(det_cls{k}, 'Car') == 1
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
            elseif strcmp(det_cls{k}, 'Pedestrian') == 1
                pattern = data_pedestrian.pattern{cid};
                pattern = imresize(pattern, [h w], 'nearest');
            elseif strcmp(det_cls{k}, 'Cyclist') == 1
                pattern = data_cyclist.pattern{cid};
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
            if strcmp(det_cls{k}, 'Car') == 1
                index_color = 1 + floor((k-1) * size(cmap,1) / num);
                dispColor = 255*cmap(index_color,:);
            else
                dispColor = 255*[.7 .5 0];
            end
            scale = round(max(size(I))/300);            
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
        if dets(k,6) > threshold && strcmp(det_cls{k}, 'Car') == 0
            % get predicted bounding box
            bbox_pr = dets(k,1:4);
            bbox_draw = [bbox_pr(1), bbox_pr(2), bbox_pr(3)-bbox_pr(1), bbox_pr(4)-bbox_pr(2)];
            if strcmp(det_cls{k}, 'Pedestrian') == 1
                rectangle('Position', bbox_draw, 'EdgeColor', 'm', 'LineWidth', 4);
            else
                rectangle('Position', bbox_draw, 'EdgeColor', [.7 .5 0], 'LineWidth', 4);
            end
        end
    end
    hold off;

    ind_plot = ind_plot + 2;
    
    % plot 3D localization
    Vpr = [];
    Fpr = [];
    Vpr_pedestrian = [];
    Fpr_pedestrian = [];
    Vpr_cyclist = [];
    Fpr_cyclist = [];    
    for k = 1:numel(objects);
        object = objects(k);
        if strcmp(object.type, 'Car') == 1 && object.score >= threshold
            % transfer cad model
            cad_index = data.cad_index(objects(k).cid);
            x3d = compute_3d_points(cads(cad_index).vertices, object);
            face = cads(cad_index).faces;
            tmp = face + size(Vpr,2);
            Fpr = [Fpr; tmp];        
            Vpr = [Vpr x3d];
        elseif strcmp(object.type, 'Pedestrian') == 1 && object.score >= threshold
            x3d = compute_3d_points(vertices_pedestrian, object);
            face = faces_pedestrian;
            tmp = face + size(Vpr_pedestrian,2);
            Fpr_pedestrian = [Fpr_pedestrian; tmp];        
            Vpr_pedestrian = [Vpr_pedestrian x3d];
        elseif strcmp(object.type, 'Cyclist') == 1 && object.score >= threshold
            x3d = compute_3d_points(vertices_cyclist, object);
            face = faces_cyclist;
            tmp = face + size(Vpr_cyclist,2);
            Fpr_cyclist = [Fpr_cyclist; tmp];        
            Vpr_cyclist = [Vpr_cyclist x3d];
        end
    end    
    
    subplot(mplot, nplot, [3, 4]);
    cla;
    hold on;
    axis equal;    
    
    
    Vpr_all = [Vpr Vpr_pedestrian Vpr_cyclist];
    if isempty(Vpr_all) == 0
        Vpr_all = Pv2c\[Vpr_all; ones(1,size(Vpr_all,2))];
        if isempty(Vpr) == 0
            Vpr = Pv2c\[Vpr; ones(1,size(Vpr,2))];
            trimesh(Fpr, Vpr(1,:), Vpr(2,:), Vpr(3,:), 'EdgeColor', 'b');
        end

        if isempty(Vpr_pedestrian) == 0
            Vpr_pedestrian = Pv2c\[Vpr_pedestrian; ones(1,size(Vpr_pedestrian,2))];
            trimesh(Fpr_pedestrian, Vpr_pedestrian(1,:), Vpr_pedestrian(2,:), Vpr_pedestrian(3,:), 'EdgeColor', 'k');
        end
        
        if isempty(Vpr_cyclist) == 0
            Vpr_cyclist = Pv2c\[Vpr_cyclist; ones(1,size(Vpr_cyclist,2))];
            trimesh(Fpr_cyclist, Vpr_cyclist(1,:), Vpr_cyclist(2,:), Vpr_cyclist(3,:), 'EdgeColor', 'k');
        end        
                
%         xlabel('x');
%         ylabel('y');
%         zlabel('z');

        % draw the camera
%         draw_camera(C);

        % draw the ground plane
        h = 1.73;
        
        sxmin = min(min(Vpr_all(1,:)), C(1)) - 5;
        symin = min(min(Vpr_all(2,:)), C(2)) - 5;
        sxmax = max(max(Vpr_all(1,:)), C(1)) + 5;
        symax = max(max(Vpr_all(2,:)), C(2)) + 5;
        plane_vertex = zeros(4,3);
        plane_vertex(1,:) = [sxmin symin -h];
        plane_vertex(2,:) = [sxmax symin -h];
        plane_vertex(3,:) = [sxmax symax -h];
        plane_vertex(4,:) = [sxmin symax -h];
        
        patch('Faces', [1 2 3 4], 'Vertices', plane_vertex, 'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.5);

        axis tight;
        % title('3D Estimation');
        view(250, 8);
        hold off;
    end
    
    ind_plot = ind_plot + 2;
    if ind_plot > mplot*nplot
        ind_plot = 1;
        if is_save
            filename = fullfile('result_images_test', sprintf('%06d.png', img_idx));
            hgexport(hf, filename, hgexport('factorystyle'), 'Format', 'png');
        else
            pause;
        end
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