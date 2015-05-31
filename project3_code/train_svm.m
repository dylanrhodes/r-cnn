function [model] = train_svm(pos, neg)
% Trains an SVM to classify the input samples

labels = [ones([size(pos, 1) 1]); zeros([size(neg, 1) 1])];
data = [pos; neg];

shuffle = randperm(size(data, 1));
data = data(shuffle, :);
labels = labels(shuffle, :);

tic
model = fitcsvm(data, labels);
toc

pred = predict(model, data);
conf_mat = confusionmat(labels, pred)
end

