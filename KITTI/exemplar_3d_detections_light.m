function exemplar_3d_detections_light

% matlabpool open;

threshold = 0.5;
cls = 'car';

% read detection results
result_dir = 'test_results_5';
filename = sprintf('%s/detections.txt', result_dir);
[ids_det, cls_det, x1_det, y1_det, x2_det, y2_det, cid_det, score_det] = ...
    textread(filename, '%s %s %f %f %f %f %d %f');
fprintf('load detection done\n');

% KITTI path
exemplar_globals;
root_dir = KITTIroot;
data_set = 'testing';
cam = 2;
calib_dir = fullfile(root_dir, [data_set '/calib']);

% load data
filename = fullfile(SLMroot, 'KITTI/data_kitti.mat');
object = load(filename);
data = object.data;
centers = unique(data.idx_ap);
centers(centers == -1) = [];
fprintf('%d clusters for car\n', numel(centers));

filename = fullfile(SLMroot, 'KITTI/data_kitti_pedestrian.mat');
object = load(filename);
data_pedestrian = object.data;
centers_pedestrian = unique(data_pedestrian.idx_pose);
centers_pedestrian(centers_pedestrian == -1) = [];
fprintf('%d clusters for pedestrian\n', numel(centers_pedestrian));

filename = fullfile(SLMroot, 'KITTI/data_kitti_cyclist.mat');
object = load(filename);
data_cyclist = object.data;
centers_cyclist = unique(data_cyclist.idx_pose);
centers_cyclist(centers_cyclist == -1) = [];
fprintf('%d clusters for cyclist\n', numel(centers_cyclist));

% compute statistics
lmean = mean(data.l);
hmean = mean(data.h);
wmean = mean(data.w);
dmin = min(data.distance);
dmean = mean(data.distance);
dmax = max(data.distance);

lmean_pedestrian = mean(data_pedestrian.l);
hmean_pedestrian = mean(data_pedestrian.h);
wmean_pedestrian = mean(data_pedestrian.w);
dmin_pedestrian = min(data_pedestrian.distance);
dmean_pedestrian = mean(data_pedestrian.distance);
dmax_pedestrian = max(data_pedestrian.distance);

lmean_cyclist = mean(data_cyclist.l);
hmean_cyclist = mean(data_cyclist.h);
wmean_cyclist = mean(data_cyclist.w);
dmin_cyclist = min(data_cyclist.distance);
dmean_cyclist = mean(data_cyclist.distance);
dmax_cyclist = max(data_cyclist.distance);

% load the mean CAD model
filename = fullfile(SLMroot, sprintf('Geometry/%s_mean.mat', cls));
object = load(filename);
cad = object.(cls);
vertices = cad.x3d;

filename = '../CAD/pedestrian.off';
[vertices_pedestrian, ~] = load_off_file(filename);

filename = '../CAD/cyclist.off';
[vertices_cyclist, ~] = load_off_file(filename);

% read ids of validation images
object = load('kitti_ids_new.mat');
ids = object.ids_test;
N = numel(ids);

dets_3d = cell(1, N);

% for each image
for i = 1:N
    img_idx = ids(i);
    tic;
    
    % get predicted bounding box
    index = strcmp(sprintf('%06d', img_idx), ids_det);
    det_cls = cls_det(index);
    det = [x1_det(index), y1_det(index), x2_det(index), y2_det(index), cid_det(index), score_det(index)];
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
    
    T = zeros(3, num);
    fprintf('image %d, %d detections\n', i, num);
    
    % projection matrix
    P = readCalibration(calib_dir, img_idx, cam);
    
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
    
    % backproject each detection into 3D
    objects = [];
    for k = 1:num
        
        % get predicted bounding box
        bbox = det(k,1:4);
        w = bbox(3) - bbox(1) + 1;
        h = bbox(4) - bbox(2) + 1;
        
        if strcmp(det_cls{k}, 'Car') == 1
            cid = centers(det(k,5));
        elseif strcmp(det_cls{k}, 'Pedestrian') == 1
            cid = centers_pedestrian(det(k,5) - numel(centers));
        elseif strcmp(det_cls{k}, 'Cyclist') == 1
            cid = centers_cyclist(det(k,5) - numel(centers) - numel(centers_pedestrian));
        end
        
        objects(k).type = det_cls{k};
        objects(k).x1 = bbox(1);
        objects(k).y1 = bbox(2);
        objects(k).x2 = bbox(3);
        objects(k).y2 = bbox(4);        
        objects(k).cid = cid;
        objects(k).score = det(k,6);
        % apply the 2D occlusion mask to the bounding box
        % check if truncated pattern
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
                objects(k).truncation = data.truncation(cid);
                objects(k).occlusion = 0;
            else
                cx = (bbox(1) + bbox(3)) / 2;
                cy = (bbox(2) + bbox(4)) / 2;
                width = w;
                height = h;
                objects(k).truncation = 0;
                occ_per = data.occ_per(cid);
                if occ_per > 0.5
                    objects(k).occlusion = 2;
                elseif occ_per > 0
                    objects(k).occlusion = 1;
                else
                    objects(k).occlusion = 0;
                end
            end
            objects(k).occ_per = data.occ_per(cid);
            objects(k).pattern = imresize(pattern, [h w], 'nearest');
        else
            cx = (bbox(1) + bbox(3)) / 2;
            cy = (bbox(2) + bbox(4)) / 2;
            width = w;
            height = h;
            if strcmp(det_cls{k}, 'Pedestrian') == 1
                objects(k).truncation = data_pedestrian.truncation(cid);
                objects(k).occlusion = data_pedestrian.occlusion(cid);
                objects(k).occ_per = data_pedestrian.occ_per(cid);
                objects(k).pattern = [];
            else
                objects(k).truncation = data_cyclist.truncation(cid);
                objects(k).occlusion = data_cyclist.occlusion(cid);
                objects(k).occ_per = data_cyclist.occ_per(cid);
                objects(k).pattern = [];
            end
        end

        % backprojection
        c = [cx; cy + height/2; 1];
        X = pinv(P) * c;
        X = X ./ X(4);
        if X(3) < 0
            X = -1 * X;
        end
        theta = atan(X(1)/X(3));
        % transform to velodyne space
        X = Pv2c\X;
        X(4) = [];
        % compute the ray
        X = X - C;
        % normalization
        X = X ./ norm(X);     

        % optimization to search for 3D bounding box
        % compute 3D points without translation
        if strcmp(det_cls{k}, 'Car') == 1
            alpha = data.azimuth(cid) + 90;
        elseif strcmp(det_cls{k}, 'Pedestrian') == 1
            alpha = data_pedestrian.azimuth(cid) + 90;
        elseif strcmp(det_cls{k}, 'Cyclist') == 1
            alpha = data_cyclist.azimuth(cid) + 90;
        end
        ry = alpha*pi/180 + theta;
        while ry > pi
            ry = ry - 2*pi;
        end
        while ry < -pi
            ry = ry + 2*pi;
        end
        if strcmp(det_cls{k}, 'Car') == 1
            x3d = compute_3d_points(vertices, lmean, hmean, wmean, ry, [0; 0; 0]);
        elseif strcmp(det_cls{k}, 'Pedestrian') == 1    
            x3d = compute_3d_points(vertices_pedestrian, lmean_pedestrian, hmean_pedestrian, wmean_pedestrian, ry, [0; 0; 0]);
        elseif strcmp(det_cls{k}, 'Cyclist') == 1
            x3d = compute_3d_points(vertices_cyclist, lmean_cyclist, hmean_cyclist, wmean_cyclist, ry, [0; 0; 0]);
        end
        
        % optimize
        options = optimset('Algorithm', 'interior-point', 'Display', 'off');
        if strcmp(det_cls{k}, 'Car') == 1
            % initialization
            x = dmean;
            % compute lower bound and upper bound
            lb = dmin;
            ub = dmax;         
            x = fmincon(@(x)compute_error(x, x3d, C, X, P, Pv2c, width, height, hmean),...
                x, [], [], [], [], lb, ub, [], options);
        elseif strcmp(det_cls{k}, 'Pedestrian') == 1
            x = dmean_pedestrian;
            lb = dmin_pedestrian;
            ub = dmax_pedestrian; 
            x = fmincon(@(x)compute_error(x, x3d, C, X, P, Pv2c, width, height, hmean_pedestrian),...
                x, [], [], [], [], lb, ub, [], options);
        elseif strcmp(det_cls{k}, 'Cyclist') == 1
            x = dmean_cyclist;
            lb = dmin_cyclist;
            ub = dmax_cyclist;
            x = fmincon(@(x)compute_error(x, x3d, C, X, P, Pv2c, width, height, hmean_cyclist),...
                x, [], [], [], [], lb, ub, [], options);
        end

        % compute the translation in camera coordinate
        t = C + x.*X;
        T(:,k) = t;
        t(4) = 1;
        t = Pv2c*t;
        
        % compute alpha
        alpha = alpha*pi/180;
        while alpha > pi
            alpha = alpha - 2*pi;
        end
        while alpha < -pi
            alpha = alpha + 2*pi;
        end         
        
        % assign variables
        objects(k).alpha = alpha;
        if strcmp(det_cls{k}, 'Car') == 1
            objects(k).l = lmean;
            objects(k).h = hmean;
            objects(k).w = wmean;
        elseif strcmp(det_cls{k}, 'Pedestrian') == 1
            objects(k).l = lmean_pedestrian;
            objects(k).h = hmean_pedestrian;
            objects(k).w = wmean_pedestrian;            
        elseif strcmp(det_cls{k}, 'Cyclist') == 1
            objects(k).l = lmean_cyclist;
            objects(k).h = hmean_cyclist;
            objects(k).w = wmean_cyclist;            
        end
        objects(k).ry = ry;
        objects(k).t = t(1:3)';
        objects(k).T = T(:,k)';
    end
    
    dets_3d{i} = objects;    
    
    toc;
end

filename = sprintf('%s/dets_3d.mat', result_dir);
save(filename, 'dets_3d', '-v7.3');

% matlabpool close;


% compute the projection error between 3D bbox and 2D bbox
function error = compute_error(x, x3d, C, X, P, Pv2c, bw, bh, h)

% compute the translate of the 3D bounding box
t = C + x .* X;
t = Pv2c*[t; 1];
t(4) = [];

% compute 3D points
x3d(1,:) = x3d(1,:) + t(1);
x3d(2,:) = x3d(2,:) + t(2) - h/2;
x3d(3,:) = x3d(3,:) + t(3);

% project the 3D bounding box into the image plane
x2d = projectToImage(x3d, P);

% compute bounding box width and height
width = max(x2d(1,:)) - min(x2d(1,:));
height = max(x2d(2,:)) - min(x2d(2,:));

% compute error
error = (width - bw)^2 + (height - bh)^2;


function x3d = compute_3d_points(vertices, l, h, w, ry, t)

x3d = vertices';

% rotation matrix to transform coordinate systems
Rx = [1 0 0; 0 0 -1; 0 1 0];
Ry = [cos(-pi/2) 0 sin(-pi/2); 0 1 0; -sin(-pi/2) 0 cos(-pi/2)];
x3d = Ry*Rx*x3d;

% scaling factors
sx = l / (max(x3d(1,:)) - min(x3d(1,:)));
sy = h / (max(x3d(2,:)) - min(x3d(2,:)));
sz = w / (max(x3d(3,:)) - min(x3d(3,:)));
x3d = diag([sx sy sz]) * x3d;

% compute rotational matrix around yaw axis
R = [+cos(ry), 0, +sin(ry);
        0, 1,       0;
     -sin(ry), 0, +cos(ry)];

% rotate and translate 3D bounding box
x3d = R*x3d;
x3d(1,:) = x3d(1,:) + t(1);
x3d(2,:) = x3d(2,:) + t(2);
x3d(3,:) = x3d(3,:) + t(3);