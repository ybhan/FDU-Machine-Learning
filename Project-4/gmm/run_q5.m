% Completed by Yuanbo Han, Dec. 26, 2017.
load digits;

errorTrain = zeros(1, 4);
errorValidation = zeros(1, 4);
errorTest = zeros(1, 4);
numComponent = [2, 5, 15, 25];
maxIter = 20; repeat = 20;

for i = 1 : 4
    K = numComponent(i);
    for r = 1:repeat
        % Train a MoG model with K components for digit 2
        %-------------------- Add your code here --------------------
        [p2,mu2,vary2,~] = mogEM_kmeans(train2, K, maxIter, 0.01, 0);
        
        % Train a MoG model with K components for digit 3
        %-------------------- Add your code here --------------------
        [p3,mu3,vary3,~] = mogEM_kmeans(train3, K, maxIter, 0.01, 0);
        
        % Caculate the probability P(d=1|x) and P(d=2|x),
        % classify examples, and compute the error rate
        % Hints: you may want to use mogLogProb function
        %-------------------- Add your code here --------------------
        [inputs_train, inputs_valid, inputs_test, ...
            target_train, target_valid, target_test] = load_data();
        
        P2GivenTrain = mogLogProb(p2,mu2,vary2,inputs_train);
        P3GivenTrain = mogLogProb(p3,mu3,vary3,inputs_train);
        train_label = (P3GivenTrain > P2GivenTrain);
        errorTrain(i) = errorTrain(i) + 1/repeat * ...
            sum(train_label ~= target_train) / length(inputs_train);
        
        P2GivenValid = mogLogProb(p2,mu2,vary2,inputs_valid);
        P3GivenValid = mogLogProb(p3,mu3,vary3,inputs_valid);
        valid_label = (P3GivenValid > P2GivenValid);
        errorValidation(i) = errorValidation(i) + 1/repeat * ...
            sum(valid_label ~= target_valid) / length(inputs_valid);
        
        P2GivenTest = mogLogProb(p2,mu2,vary2,inputs_test);
        P3GivenTest = mogLogProb(p3,mu3,vary3,inputs_test);
        test_label = (P3GivenTest > P2GivenTest);
        errorTest(i) = errorTest(i) + 1/repeat * ...
            sum(test_label ~= target_test) / length(inputs_test);
    end
end

% Plot the error rate
%-------------------- Add your code here ----------------------------
figure;
hold on;
plot(1:4, errorTrain, 'LineWidth', 2);
plot(1:4, errorValidation, 'LineWidth', 2);
plot(1:4, errorTest, 'LineWidth', 2);
set(gca, 'XTick', 1:4);
set(gca, 'XTickLabel', numComponent);
lgd = legend({'Train','Validation','Test'}, 'Location', 'NorthEast');
set(lgd, 'FontSize', 12);
xlabel('Number of Components', 'FontSize', 12);
ylabel('Error Rate', 'FontSize', 12);
