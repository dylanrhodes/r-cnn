function train_rcnn()

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

all_positives = get_positive_features();

models = cell(NUM_CLASSES, 1);

for class = 1:NUM_CLASSES,
    pos = all_positives{class};
    neg = sample_neg_features(class, 1000);
    
    model = models{class};
    
    if model == 3 % Is new model
        % Train on things
    else
        % Score neg, retrain on highest scores
    end
end