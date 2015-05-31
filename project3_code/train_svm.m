function [model] = train_svm(pos, neg)
% Trains an SVM to classify the input samples

model = struct;
labels = [ones([size(pos, 1) 1]); zeros([size(neg, 1) 1])];
data = [pos; neg];

shuffle = randperm(size(data, 1));
data = sparse(data(shuffle, :));
labels = labels(shuffle, :);

tic
model.svm = train(labels, data, '-w1 2 -c 1e-3 -s 3 -B 10');
toc

[pred, acc, scores] = predict(labels, data, model.svm);
conf_mat = confusionmat(labels, pred)
end

