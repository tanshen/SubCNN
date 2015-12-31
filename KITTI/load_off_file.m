% load an off file
function [vertices, faces] = load_off_file(filename)

vertices = [];
faces = [];

fid = fopen(filename, 'r');
line = fgetl(fid);
if strcmp(line, 'OFF') == 0
    fprintf('Wrong format .off file %s!\n', filename);
    return;
end

line = fgetl(fid);
num = sscanf(line, '%f', 3);
nv = num(1);
nf = num(2);
vertices = zeros(nv, 3);
faces = zeros(nf, 3);

for i = 1:nv
    line = fgetl(fid);
    vertices(i,:) = sscanf(line, '%f', 3);
end

for i = 1:nf
    line = fgetl(fid);
    fsize = sscanf(line, '%f', 1);
    if fsize ~= 3
        printf('Face contains more than 3 vertices!');
    end
    temp = sscanf(line, '%f', 4);
    faces(i,:) = temp(2:4);
end
faces = faces + 1;

fclose(fid);