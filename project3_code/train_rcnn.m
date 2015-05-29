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

DIFFICULT_THRESHOLD = 0.9;
RETRAIN_THRESHOLD = 10000;

all_pos = get_positive_features();

im_data = load('train_ims.mat');
images = im_data.images;

models = cell(NUM_CLASSES, 1);

for class = 1:NUM_CLASSES,
    pos = all_pos{class};
    model = models{class};
    
    for img_idx = 1:numel(images)
        neg = get_negative_features(images(i).fname, i, class);
        hard_neg = [];
        
        if model == 'none' % Is new model
            labels = [ones([size(pos, 1) 1]); zeros([size(neg, 1) 1])];
            data = [pos; neg];
            
            model = svmtrain(data, labels);
            models{class} = model;
        else
            pred = svmclassify(model, neg);
            
            if size(hard_neg, 1) == 1
                hard_neg = neg(pred == 1, :);
            else
                hard_neg = [hard_neg; neg(pred == 1, :)];
            end
        end
        
        if size(hard_neg, 1) > RETRAIN_THRESHOLD
            labels = [ones([size(pos, 1) 1]); zeros([size(hard_neg, 1) 1])];
            data = [pos; hard_neg];
            
            model = svmtrain(data, labels);
            models{class} = model;
            
            pred = svmclassify(model, hard_neg);
            hard_neg = hard_neg(pred == 1, :);
        end
    end
    
end

