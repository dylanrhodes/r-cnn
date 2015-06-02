function [models] = train_rcnn(hyp)

% TODO
% Train the R-CNN based on your extracted features
% Things to be careful about:
% -What threshold should you use for hard negatives?
% -What overlap with ground truth should count as positive?
% -What overlap with ground truth should count as negative?
% -How often should the detector be re-trained?
% -Should you train all classes at once or do them one at a time?

rng(183927481);

HARD_THRESH = 1.0;

RETRAIN_THRESHOLD = 1500;
NUM_CLASSES = 3;

all_pos = get_positive_features();

im_data = load('train_ims.mat');
images = im_data.images;

models = cell(NUM_CLASSES, 1);

for class = 1:NUM_CLASSES,
    pos = all_pos{class};
    hard_neg = [];
    model = models{class};
    neg_acc = [];
    cv_acc = [];
    last_train_size = 0;
    
    for img_idx = 1:500 %numel(images)
        %fprintf('Processing negatives from %d/%d ...\n', img_idx, numel(images));
        neg = get_negative_features(images(img_idx), img_idx, class);
        
        % Initialize new model
        if isempty(model)
            [model, cv_acc] = train_svm(pos, neg, hyp);
            cv_acc = [1 cv_acc];
            models{class} = model;
        end
        
        [pred, acc, scores] = predict(zeros(size(neg, 1), 1), sparse(prep(neg, model)), model.svm, '-q');
        neg_acc = [neg_acc; img_idx acc(1)];
        
        if min(scores(pred == 0)) < 0
            hard_neg = [hard_neg; neg(scores > -HARD_THRESH, :)];
        else
            hard_neg = [hard_neg; neg(scores < HARD_THRESH, :)];
        end
        
        %fprintf('Adding %.0f negative samples... \n', sum(scores < HARD_THRESH));
        
        % Retrain SVM's when we have accumulated enough new hard negatives
        if size(hard_neg, 1) - last_train_size > RETRAIN_THRESHOLD
            [model, hard_cv_acc] = train_svm(pos, hard_neg, hyp);
            cv_acc = [cv_acc; img_idx hard_cv_acc];
            models{class} = model;
            
            [pred, acc, scores] = predict(zeros(size(hard_neg, 1), 1), sparse(prep(hard_neg, model)), model.svm);
            
            if min(scores(pred == 0)) < 0
                hard_neg = hard_neg(scores > -HARD_THRESH, :);
            else
                hard_neg = hard_neg(scores < HARD_THRESH, :);
            end
            
            last_train_size = size(hard_neg, 1);
            fprintf('Retaining %d hard negatives...\n', size(hard_neg, 1));
        end
    end
    
    %figure
    %hold on;
    %title(sprintf('Class: %d, Weight: %d, Cost: %.5f, Bias:%d', [class hyp]));
    %plot(neg_acc(:,1), neg_acc(:,2), 'r-', cv_acc(:, 1), cv_acc(:, 2), 'g-');
    %legend('Neg. Acc.', 'Hard CV acc.');
    %hold off;
end

