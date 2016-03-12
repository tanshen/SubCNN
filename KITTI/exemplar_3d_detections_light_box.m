function exemplar_3d_detections_light_box

matlabpool open;

% threshold = 0.5;
cls = 'car';
is_train = 1;

% read ids of validation images
object = load('kitti_ids_new.mat');
if is_train
    ids = object.ids_val;
else
    ids = object.ids_test;
end
N = numel(ids);

% read detection results
if is_train
    result_dir = 'results_faster_rcnn';
else
    result_dir = 'test_results_5';
end
detections = cell(1, N);
classes = cell(1, N);
parfor i = 1:N
    filename = sprintf('%s/%06d.txt', result_dir, ids(i));
    disp(filename);
    fid = fopen(filename, 'r');
    C = textscan(fid, '%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f');   
    fclose(fid);
    
    det = double([C{5} C{6} C{7} C{8} C{end}]);
    index = strcmp('Car', C{1});
    det = det(index, :);
    detections{i} = det;
    classes{i} = C{1}(index);
end
fprintf('load detection done\n');

% KITTI path
exemplar_globals;
root_dir = KITTIroot;
data_set = 'testing';
cam = 2;
calib_dir = fullfile(root_dir, [data_set '/calib']);

% load data
if is_train
    filename = fullfile(SLMroot, 'KITTI/data.mat');
else
    filename = fullfile(SLMroot, 'KITTI/data_kitti.mat');
end
object = load(filename);
data = object.data;
centers = unique(data.idx_ap);
centers(centers == -1) = [];
fprintf('%d clusters for car\n', numel(centers));

if is_train
    filename = fullfile(SLMroot, 'KITTI/data_pedestrian.mat');
else
    filename = fullfile(SLMroot, 'KITTI/data_kitti_pedestrian.mat');
end
object = load(filename);
data_pedestrian = object.data;
centers_pedestrian = unique(data_pedestrian.idx_pose);
centers_pedestrian(centers_pedestrian == -1) = [];
fprintf('%d clusters for pedestrian\n', numel(centers_pedestrian));

if is_train
    filename = fullfile(SLMroot, 'KITTI/data_cyclist.mat');
else
    filename = fullfile(SLMroot, 'KITTI/data_kitti_cyclist.mat');
end
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

dets_3d = cell(1, N);

% for each image
parfor i = 1:N
    img_idx = ids(i);
    tic;
    
    % get predicted bounding box
    det = detections{i};
    det_cls = classes{i};
    if isempty(det) == 1
        fprintf('no detection for image %d\n', img_idx);
        continue;
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
        
        objects(k).type = det_cls{k};
        objects(k).x1 = bbox(1);
        objects(k).y1 = bbox(2);
        objects(k).x2 = bbox(3);
        objects(k).y2 = bbox(4);
        objects(k).score = det(k,5);
        % apply the 2D occlusion mask to the bounding box
        % check if truncated pattern

        cx = (bbox(1) + bbox(3)) / 2;
        cy = (bbox(2) + bbox(4)) / 2;
        width = w;
        height = h;
        objects(k).truncation = 0;
        occ_per = 0;
        if occ_per > 0.5
            objects(k).occlusion = 2;
        elseif occ_per > 0
            objects(k).occlusion = 1;
        else
            objects(k).occlusion = 0;
        end
        objects(k).occ_per = 0;
        objects(k).pattern = [];

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
        alpha = 0;
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

matlabpool close;


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