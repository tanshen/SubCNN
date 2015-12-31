% compute the 3D point locations of a CAD model
function x3d = compute_3d_points(vertices, object)

x3d = vertices';

% rotation matrix to transform coordinate systems
Rx = [1 0 0; 0 0 -1; 0 1 0];
Ry = [cos(-pi/2) 0 sin(-pi/2); 0 1 0; -sin(-pi/2) 0 cos(-pi/2)];
x3d = Ry*Rx*x3d;

% scaling factors
sx = object.l / (max(x3d(1,:)) - min(x3d(1,:)));
sy = object.h / (max(x3d(2,:)) - min(x3d(2,:)));
sz = object.w / (max(x3d(3,:)) - min(x3d(3,:)));
x3d = diag([sx sy sz]) * x3d;

% compute rotational matrix around yaw axis
R = [+cos(object.ry), 0, +sin(object.ry);
                   0, 1,               0;
     -sin(object.ry), 0, +cos(object.ry)];

% rotate and translate 3D bounding box
x3d = R*x3d;
x3d(1,:) = x3d(1,:) + object.t(1);
x3d(2,:) = x3d(2,:) + object.t(2) - object.h/2;
x3d(3,:) = x3d(3,:) + object.t(3);