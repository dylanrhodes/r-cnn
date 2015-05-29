function [output] = get_positive_features()
% Loads feature vectors for all ground-truth bounding boxes

im_data = load('train_ims.mat');
images = im_data.images;

output = cell(3, 1);

for i = 1:numel(images)
    fname = ['../features/' images(i).fname(1:size(images(i).fname, 2) - 4) '.mat']
    in_file = fopen(fname, 'r');
    cnn_feat = fread(in_file, 'single');
    fclose(in_file);
    
    for j = 1:size(images(i).classes, 2)
        class = images(i).classes(1, j)
        
        if size(output{class}, 2) == 1
            output{class} = cnn_feat(j, :);
        else
            output{class} = [output{class}; cnn_feat(j, :)];
        end
    end
end
end

