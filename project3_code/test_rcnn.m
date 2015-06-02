function [res] = test_rcnn(models)

% TODO
% Run your trained R-CNN on all test images
% Do non-max suppression
% Save and evaluate your predictions.

NUM_CLASSES = 3;

im_data = load('train_ims.mat');
images = im_data.images;
images = images(500:750);

ssearch_data = load('ssearch_train.mat');
ssearch_boxes = ssearch_data.ssearch_boxes;
ssearch_boxes = ssearch_boxes(500:750);

pred_bboxes = cell(numel(images), NUM_CLASSES);
gt_bboxes = cell(numel(images), NUM_CLASSES);

for img_idx = 1:numel(images)
    fname = ['../features/' images(img_idx).fname(1:size(images(img_idx).fname, 2) - 4) '.bin'];
    feat = read_cnn_features(fname);
    feat = feat(1 + length(images(img_idx).classes):size(feat, 1), :);
    boxes = ssearch_boxes{img_idx};
    
    for class = 1:NUM_CLASSES
        [pred, acc, score] = predict(zeros(size(feat, 1), 1), sparse(prep(feat, models{class})), models{class}.svm);
        
        pred_boxes = boxes(pred == 1, :);
        pred_scores = score(pred == 1, :);
        
        [pred_boxes, pred_scores] = non_max_suppression(pred_boxes, pred_scores);
        % TODO: ridge regression for bboxes
        
        gt_bboxes{img_idx, class} = images(img_idx).bboxes(images(img_idx).classes == class, :);
        pred_bboxes{img_idx, class} = [pred_boxes pred_scores];
    end
end

res = zeros(NUM_CLASSES, 3);

for class = 1:NUM_CLASSES
    [ap, prec, rec] = det_eval(pred_bboxes(:, class), gt_bboxes(:, class));
    res(class, :) = [ap mean(prec) mean(rec)];
end

res
