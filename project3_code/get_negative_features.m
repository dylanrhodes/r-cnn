function [feat] = get_negative_features(img, img_idx, class)
% Return features for the bounding boxes which are negative for this class

NEGATIVE_THRESHOLD = 0.3;

ssearch_data = load('ssearch_train.mat');
ssearch_boxes = ssearch_data.ssearch_boxes;

fname = ['../features/' img.fname(1:size(img.fname, 2) - 4) '.mat']
in_file = fopen(fname, 'r');
feat = fread(in_file, 'single');
fclose(in_file);

num_rows = size(feat, 1) / 512;
feat = reshape(feat, [num_rows 512]);

pos_boxes = img.bboxes(img.classes == class, :);
neg_boxes = ssearch_boxes(i);

for i = 1:size(pos_boxes, 1)
    overlaps = boxoverlap(neg_boxes, pos_boxes(i, :));
    neg_boxes = neg_boxes(overlaps < NEGATIVE_THRESHOLD, :);
    feat = feat(overlaps < NEGATIVE_THRESHOLD, :);
end

end

