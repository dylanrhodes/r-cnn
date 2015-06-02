function [data] = prep(data, model)
% Preprocess data according to settings in model

data = (data - repmat(model.mean, size(data, 1), 1)) ./ repmat(model.std, size(data, 1), 1);

end

