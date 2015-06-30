function write_dets_to_files

% load ids
object = load('kitti_ids_new.mat');
ids_train = object.ids_train;
ids_val = object.ids_val;
ids_test = object.ids_test;

% load detections
object = load('car_3d_aps_125_combined_train.mat');
dets_train = object.dets;
object = load('car_3d_aps_125_combined_test.mat');
dets_val = object.dets;

for i = 1:numel(ids_train)
    filename = sprintf('3DVP_125/training/%06d.txt', ids_train(i));
    disp(filename);
    fid = fopen(filename, 'w');

    det = dets_train{i};
    for j = 1:size(det,1)
        fprintf(fid, '%f %f %f %f\n', det(j,1), det(j,2), det(j,3), det(j,4));
    end

    fclose(fid);
end

for i = 1:numel(ids_val)
    filename = sprintf('3DVP_125/training/%06d.txt', ids_val(i));
    disp(filename);
    fid = fopen(filename, 'w');

    det = dets_val{i};
    for j = 1:size(det,1)
        fprintf(fid, '%f %f %f %f\n', det(j,1), det(j,2), det(j,3), det(j,4));
    end

    fclose(fid);
end
