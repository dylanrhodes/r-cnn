function test_rcnn()

% TODO
% Run your trained R-CNN on all test images
% Do non-max suppression
% Save and evaluate your predictions.

NUM_CLASSES = 1;
DETECTION_THRESH = 5;

im_data = load('test_ims.mat');
images = im_data.images;

ssearch_data = load('ssearch_test.mat');
ssearch_boxes = ssearch_data.ssearch_boxes;

pred_bboxes = cell(numel(images), NUM_CLASSES);
gt_bboxes = cell(numel(images), NUM_CLASSES);

models = train_rcnn()

for img_idx = 1:numel(images)
    fname = ['../features/' images(img_idx).fname(1:size(images(img_idx).fname, 2) - 4) '.bin'];
    feat = read_cnn_features(fname);
    boxes = ssearch_boxes{img_idx};
    
    for class = 1:NUM_CLASSES
        [pred, score] = predict(models{class}.svm, feat);
        
        pred_boxes = boxes(score(:,2) > DETECTION_THRESH, :);
        pred_scores = score(score(:,2) > DETECTION_THRESH, 2);
        
        [pred_boxes, pred_scores] = non_max_suppression(pred_boxes, pred_scores);
        % TODO: ridge regression for bboxes
        
        gt_bboxes{img_idx, class} = images(img_idx).bboxes(images(img_idx).classes == class, :);
        pred_bboxes{img_idx, class} = [pred_boxes pred_scores];
    end
end

for class = 1:NUM_CLASSES
    det_eval(pred_bboxes(:, class), gt_bboxes(:, class))
end