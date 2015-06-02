function [model, cv_acc] = train_svm(pos, neg, hyp)
% Trains an SVM to classify the input samples

% Previous best hyperparameters
% SVM_PARAMS = '-w1 10 -c 1e-3 -s 3 -B 1 -q ';

% Hyperparameters to test
SVM_PARAMS = sprintf('-w1 %d -c %.6f -s 3 -B %d -q ', hyp);
fprintf('SVM PARAMS: %s \n', SVM_PARAMS);

model = struct;
labels = [ones([size(pos, 1) 1]); zeros([size(neg, 1) 1])];
data = [pos; neg];

shuffle = randperm(size(data, 1));
data = sparse(data(shuffle, :));
labels = labels(shuffle, :);

model.mean = mean(data);
model.std = std(data);
data = prep(data, model);

%cv_acc = train(labels, data, [SVM_PARAMS '-v 2 ']);
cv_acc = 0.0;
model.svm = train(labels, data, SVM_PARAMS);

[pred, acc, scores] = predict(labels, data, model.svm);
conf_mat = confusionmat(labels, pred)
end

