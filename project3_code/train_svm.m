function [model] = train_svm(pos, neg)
% Trains an SVM to classify the input samples

model = struct;
labels = [ones([size(pos, 1) 1]); zeros([size(neg, 1) 1])];
data = [pos; neg];

shuffle = randperm(size(data, 1));
data = single(data(shuffle, :));
labels = labels(shuffle, :);

tic
model.svm = fitcsvm(data, labels, 'BoxConstraint', 10.0, 'ClassNames', [0, 1], 'Cost', [0 5; 1 0], 'Standardize', true);
toc

pred = predict(model.svm, data);
conf_mat = confusionmat(labels, pred)
end

