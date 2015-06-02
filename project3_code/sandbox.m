function sandbox()
% This function trains and evaluates RCNN's with different hyperparameters
% on a validation set (the last third of the training images)

weights = [2 5 10 20];
costs = [1e-2 1e-3 1e-4 1e-5];
biases = [1 10];

results = cell(length(weights), length(costs), length(biases));
mean_results = cell(length(weights), length(costs), length(biases));

for w = 1:length(weights)
    for c = 1:length(costs)
        for b = 1:length(biases)
            models = train_rcnn([weights(w) costs(c) biases(b)]);
            conf_res = test_rcnn(models);
            
            results{w, c, b} = conf_res;
            mean_results{w, c, b} = mean(conf_res);
        end
    end
end

disp(results);

end

