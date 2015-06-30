function write_ids_to_files

object = load('kitti_ids_new.mat');
ids_train = object.ids_train;
ids_val = object.ids_val;
ids_test = object.ids_test;

filename = 'train.txt';
fid = fopen(filename, 'w');
ids = ids_train;
for i = 1:numel(ids)
    fprintf(fid, '%06d\n', ids(i));
end
fclose(fid);

filename = 'val.txt';
fid = fopen(filename, 'w');
ids = ids_val;
for i = 1:numel(ids)
    fprintf(fid, '%06d\n', ids(i));
end
fclose(fid);


filename = 'trainval.txt';
fid = fopen(filename, 'w');
ids = sort([ids_train ids_val]);
for i = 1:numel(ids)
    fprintf(fid, '%06d\n', ids(i));
end
fclose(fid);

filename = 'test.txt';
fid = fopen(filename, 'w');
ids = ids_test;
for i = 1:numel(ids)
    fprintf(fid, '%06d\n', ids(i));
end
fclose(fid);