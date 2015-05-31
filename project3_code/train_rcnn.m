function [models] = train_rcnn()

% TODO
% Train the R-CNN based on your extracted features
% Things to be careful about:
% -What threshold should you use for hard negatives?
% -What overlap with ground truth should count as positive?
% -What overlap with ground truth should count as negative?
% -How often should the detector be re-trained?
% -What type of feature normalization should you do?
% -Should you use any regularization?
% -How many epochs over the data should you do?
% -Should you train all classes at once or do them one at a time?
% -Should you use a bias?
% -What type of SVM solver/formulation should you use?

HARD_THRESH = 1.0;
RETRAIN_THRESHOLD = 2500;
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
    
    for img_idx = 1:50 %numel(images)
        fprintf('Processing negatives from %d/%d ...\n', img_idx, numel(images));
        neg = get_negative_features(images(img_idx), img_idx, class);
        
        % Initialize new model
        if isempty(model)
            model = train_svm(pos, neg);
            models{class} = model;
        end
        
        pred = predict(model, neg);
        neg_acc = [neg_acc (1-(sum(pred) / size(pred, 1)))];

        if size(hard_neg, 1) == 1
            hard_neg = neg(pred == 1, :);
        else
            hard_neg = [hard_neg; neg(pred == 1, :)];
        end
        
        if size(hard_neg, 1) > RETRAIN_THRESHOLD
            model = train_svm(pos, hard_neg);
            models{class} = model;
            
            [pred, scores] = predict(model, hard_neg);
            hard_neg = hard_neg(scores(:, 1) < HARD_THRESH, :);
            fprintf('Retaining %d hard negatives...\n', size(hard_neg, 1));
        end
    end
    
    plot(neg_acc);
    
end

