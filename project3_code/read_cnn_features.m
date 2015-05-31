function [feat] = read_cnn_features(in_file)
% Utility for reading features from binary file

in_file = fopen(in_file, 'r');
feat = fread(in_file, 'single');
fclose(in_file);

num_rows = size(feat, 1) / 512;
feat = reshape(feat, [num_rows 512]);
end

