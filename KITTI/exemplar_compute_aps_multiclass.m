function exemplar_compute_aps_multiclass

results_dir = 'results_kitti_train_googlenet';

command = sprintf('./evaluate_object %s', results_dir);
system(command);

classes = {'car', 'pedestrian', 'cyclist'};
figure(1);
for i = 1:numel(classes)
    cls = classes{i};
    
    % detection
    filename = sprintf('%s/plot/%s_detection.txt', results_dir, cls);
    data = load(filename);

    recall = data(:,1);
    precision_easy = data(:,2);
    precision_moderate = data(:,3);
    precision_hard = data(:,4);

    ap_easy = VOCap(recall, precision_easy);
    fprintf('%s AP_easy = %.4f\n', cls, ap_easy);

    ap_moderate = VOCap(recall, precision_moderate);
    fprintf('%s AP_moderate = %.4f\n', cls, ap_moderate);

    ap = VOCap(recall, precision_hard);
    fprintf('%s AP_hard = %.4f\n', cls, ap);

    % draw recall-precision and accuracy curve
    subplot(3, 2, 2*i-1);
    hold on;
    plot(recall, precision_easy, 'g', 'LineWidth',3);
    plot(recall, precision_moderate, 'b', 'LineWidth',3);
    plot(recall, precision_hard, 'r', 'LineWidth',3);
    h = xlabel('Recall');
    set(h, 'FontSize', 12);
    h = ylabel('Precision');
    set(h, 'FontSize', 12);
    tit = sprintf('%s AP', cls);
    h = title(tit);
    set(h, 'FontSize', 12);
    hold off;


    % pose estimation
    filename = sprintf('%s/plot/%s_orientation.txt', results_dir, cls);
    if exist(filename, 'file')
        data = load(filename);

        recall = data(:,1);
        precision_easy = data(:,2);
        precision_moderate = data(:,3);
        precision_hard = data(:,4);

        ap_easy = VOCap(recall, precision_easy);
        fprintf('%s AOS_easy = %.4f\n', cls, ap_easy);

        ap_moderate = VOCap(recall, precision_moderate);
        fprintf('%s AOS_moderate = %.4f\n', cls, ap_moderate);

        ap = VOCap(recall, precision_hard);
        fprintf('%s AOS_hard = %.4f\n', cls, ap);

        % draw recall-precision and accuracy curve
        subplot(3, 2, 2*i);
        hold on;
        plot(recall, precision_easy, 'g', 'LineWidth',3);
        plot(recall, precision_moderate, 'b', 'LineWidth',3);
        plot(recall, precision_hard, 'r', 'LineWidth',3);
        h = xlabel('Recall');
        set(h, 'FontSize', 12);
        h = ylabel('Precision');
        set(h, 'FontSize', 12);
        tit = sprintf('%s AOS', cls);
        h = title(tit);
        set(h, 'FontSize', 12);
        hold off;
    end
end