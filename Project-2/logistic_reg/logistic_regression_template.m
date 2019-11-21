% Completed by Yuanbo Han, 2017-11-12.
% Modified by Yuanbo Han, 2017-11-13: Delete some spare parts of codes and
%                                     an improper coefficient in line 54.

%% Clear workspace and close figures.
clear all;
close all;

%% Load data.
load mnist_train_small;
load mnist_valid;
load mnist_test;

%% TODO: Initialize hyperparameters.
% Learning rate
hyperparameters.learning_rate = 0.001;
% Weight regularization parameter
hyperparameters.weight_regularization = 0.01;
% Number of iterations
hyperparameters.num_iterations = 300;
% Logistic regression weights
% TODO: Set random weights.
weights = randn( (size(train_inputs_small,2) + 1), 1 );

%% Verify that your logistic function produces the right gradient, diff should be very close to 0.
% This creates small random data with 20 examples and 10 dimensions and checks the gradient on that data.
nexamples = 20;
ndimensions = 10;
diff = checkgrad('logistic', ...
    randn((ndimensions + 1), 1), ...   % weights
    0.001,...                          % perturbation
    randn(nexamples, ndimensions), ... % data
    rand(nexamples, 1), ...            % targets
    hyperparameters)                   % other hyperparameters

N = size(train_inputs_small,1);
%% Begin learning with gradient descent.
for t = 1:hyperparameters.num_iterations
    
    % Find the negative log likelihood and derivatives w.r.t. weights.
    [f, df, predictions] = logistic(weights, ...
        train_inputs_small, ...
        train_targets_small, ...
        hyperparameters);
    
    % Report the possible errors.
    if isnan(f) || isinf(f)
        error('nan/inf error');
    end
    
    % Find the cross entropy and fraction of correctly classified examples of training data.
    [cross_entropy_train, frac_correct_train] = evaluate(train_targets_small, predictions);
    
    %% Update weights.
    weights = weights - hyperparameters.learning_rate .* df;
    
    % Find the cross entropy and fraction of correctly classified examples of validation data.
    predictions_valid = logistic_predict(weights, valid_inputs);
    [cross_entropy_valid, frac_correct_valid] = evaluate(valid_targets, predictions_valid);
    
    % Find the cross entropy and fraction of correctly classified examples of test data.
    predictions_test = logistic_predict(weights, test_inputs);
    [cross_entropy_test, frac_correct_test] = evaluate(test_targets, predictions_test);
    
    %% Print some stats.
    fprintf(1, 'ITERATION:%4i   NLOGL:%5.2f TRAIN_CE:%.6f TRAIN_FRAC:%.2f VALIC_CE:%.6f VALID_FRAC:%.2f TEST_CE:%.6f TEST_FRAC:%.2f\n',...
        t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100, cross_entropy_test, frac_correct_test*100);
    
end

clear t;
