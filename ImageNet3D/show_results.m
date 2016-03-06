function show_results

cls = 'aeroplane';
network = 'caffenet';
region_proposal = 'selective_search';
threshold = 0.7;

opt = globals();
root = opt.path_imagenet3d;

result_dir = '/scail/scratch/u/yuxiang/3DVP_RCNN/fast-rcnn/output/imagenet3d/imagenet3d_test';
if exist(result_dir, 'dir') == 0
    result_dir = '/home/yuxiang/Projects/3DVP_RCNN/fast-rcnn/output/imagenet3d/imagenet3d_test';
end
method = sprintf('%s_fast_rcnn_view_imagenet3d_%s_iter_160000', network, region_proposal);

% load detection
filename = sprintf('%s/%s/detections_%s.txt', result_dir, method, cls);
fid = fopen(filename, 'r');
C = textscan(fid, '%s %f %f %f %f %f %f %f %f');
fclose(fid);

ids = C{1};
b1 = C{2};
b2 = C{3};
b3 = C{4};
b4 = C{5};
confidence = C{6};
BB = [b1 b2 b3 b4]';
azimuth = C{7};
elevation = C{8};
rotation = C{9};

index = confidence > threshold;
confidence = confidence(index);
BB = BB(:,index);
ids = ids(index);
azimuth = azimuth(index);
elevation = elevation(index);
rotation = rotation(index);

% sort detections by decreasing confidence
[~, si]=sort(-confidence);
ids = ids(si);
BB = BB(:,si);
azimuth = azimuth(si);
elevation = elevation(si);
rotation = rotation(si);

gtids = unique(ids);
M = numel(gtids);
gtids = gtids(randperm(M));

% for each image
for i = 1:M
    % find detections
    index = find(strcmp(gtids{i}, ids) == 1);
    
    % draw bbox
    if isempty(index) == 0
        % read image
        filename = sprintf('%s/Images/%s.JPEG', root, gtids{i});
        I = imread(filename);
        imshow(I);        
        
        ind = index(1);
        bbox = [BB(1,ind) BB(2,ind) BB(3,ind)-BB(1,ind) BB(4,ind)-BB(2,ind)];
        rectangle('Position', bbox, 'EdgeColor', 'g', 'LineWidth', 2);
        
        a = azimuth(ind) * 180 / pi;
        e = elevation(ind) * 180 / pi;
        r = rotation(ind) * 180 / pi;
        til = sprintf('azimuth %.2f, elevation %.2f, rotation %.2f', a, e, r);
        title(til);
    end
    pause;
end