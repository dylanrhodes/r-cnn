function [boxes, scores] = non_max_suppression(boxes, scores)
% Perform non-max suppression

OVERLAP_THRESH = 0.3;

[scores, idxs] = sort(scores, 'descend');
boxes = boxes(idxs, :);
curr_size = size(boxes, 1);

for i = 1:curr_size
    if i > curr_size
        continue
    end
    
    lower_boxes = boxes(i+1:size(boxes, 1), :);
    lower_scores = scores(i+1:size(scores, 1));
    
    overlaps = boxoverlap(lower_boxes, boxes(i, :));
    idxs = overlaps < OVERLAP_THRESH;
    
    boxes = [boxes(1:i, :); lower_boxes(idxs, :)];
    scores = [scores(1:i); lower_scores(idxs, :)];
    curr_size = size(boxes, 1);
end

end

