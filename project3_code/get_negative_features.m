function [feat] = get_negative_features(img, img_idx, class)
% Return features for the bounding boxes which are negative for this class

NEGATIVE_THRESHOLD = 0.3;

ssearch_data = load('ssearch_train.mat');
ssearch_boxes = ssearch_data.ssearch_boxes;

fname = ['../features/' img.fname(1:size(img.fname, 2) - 4) '.bin'];
feat = read_cnn_features(fname);
feat = feat(size(img.bboxes, 1)+1:size(feat, 1), :);

pos_boxes = img.bboxes(img.classes == class, :);
neg_boxes = ssearch_boxes{img_idx};

for i = 1:size(pos_boxes, 1)
    overlaps = boxoverlap(neg_boxes, pos_boxes(i, :));
    neg_boxes = neg_boxes(overlaps < NEGATIVE_THRESHOLD, :);
    feat = feat(overlaps < NEGATIVE_THRESHOLD, :);
end

end

