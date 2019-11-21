% Edited by Yuanbo Han, 2017-11-12.

%% Clear workspace and close figures.
clear all;
close all;

%% Load data.
load ../data/mnist_train;
load ../data/mnist_train_small;
load ../data/mnist_valid;

%% Initialize hyperparameters.
% Learning rate
hyperparameters.learning_rate = 0.001;
% Weight regularization parameter
hyperparameters.weight_regularization = 0;
% Number of iterations
hyperparameters.num_iterations = 300;
% Logistic regression weights
% Set random weights.
weights = randn( (size(train_inputs,2) + 1), 1 );
%weights = zeros( (size(train_inputs,2) + 1), 1 );
weights_small = weights;

N = size(train_inputs, 1);
N_small = size(train_inputs_small, 1);

cross_entropy_train = zeros( hyperparameters.num_iterations, 1 );
cross_entropy_train_small = cross_entropy_train;
cross_entropy_valid = cross_entropy_train;
cross_entropy_valid_small = cross_entropy_train;

%% Begin learning with gradient descent.
for t = 1:hyperparameters.num_iterations
    
    % Find the negative log likelihood and derivatives w.r.t. weights.
    [f, df, predictions] = logistic(weights, ...
        train_inputs, ...
        train_targets, ...
        hyperparameters);
    
    [f_small, df_small, predictions_small] = logistic(weights_small, ...
        train_inputs_small, ...
        train_targets_small, ...
        hyperparameters);
    
    % Report the possible errors.
    if isnan(f) || isinf(f)
        error('f nan/inf error');
    end
    
    if isnan(f_small) || isinf(f_small)
        error('f_small nan/inf error');
    end
    
    % Find the cross entropy and fraction of correctly classified examples of training data.
    [cross_entropy_train(t), frac_correct_train] = evaluate(train_targets, predictions);
    [cross_entropy_train_small(t), frac_correct_train_small] = evaluate(train_targets_small, predictions_small);
    
    % Update weights.
    weights = weights - hyperparameters.learning_rate .* df;
    weights_small = weights_small - hyperparameters.learning_rate .* df_small;
    
    % Find the cross entropy and fraction of correctly classified examples of validation data.
    predictions_valid = logistic_predict(weights, valid_inputs);
    predictions_valid_small = logistic_predict(weights_small, valid_inputs);
    
    [cross_entropy_valid(t), frac_correct_valid] = evaluate(valid_targets, predictions_valid);
    [cross_entropy_valid_small(t), frac_correct_valid_small] = evaluate(valid_targets, predictions_valid_small);
    
    % Print some stats.
    fprintf(1, 'ITERATION:%4i   NLOGL:%11.2f TRAIN_CE:%16.6f TRAIN_FRAC:%12.2f VALIC_CE:%16.6f VALID_FRAC:%12.2f\n',...
        t, f/N, cross_entropy_train(t), frac_correct_train*100, cross_entropy_valid(t), frac_correct_valid*100);
    fprintf('%17sNLOGL_SMALL:%5.2f TRAIN_SMALL_CE:%10.6f TRAIN_SMALL_FRAC:%6.2f VALID_CE_SMALL:%10.6f VALID_FRAC_SMALL:%6.2f\n',...
        '', f_small/N_small, cross_entropy_train_small(t), frac_correct_train_small*100, cross_entropy_valid_small(t), frac_correct_valid_small*100);
    
end

%% Plot the cross entropy as training progresses.
figure;
subplot(1,2,1);
hold on;
title('mnist\_train', 'FontSize', 15);
plot(1:hyperparameters.num_iterations, cross_entropy_train, 'LineWidth', 2);
plot(1:hyperparameters.num_iterations, cross_entropy_valid, 'LineWidth', 2);
lgd = legend('train', 'valid', 'Location', 'NorthEast');
set(lgd, 'FontSize', 12);
xlabel('Iteration', 'FontSize', 12);
ylabel('Cross entropy', 'FontSize', 12);

subplot(1,2,2);
hold on;
title('mnist\_train\_small', 'FontSize', 15);
plot(1:hyperparameters.num_iterations, cross_entropy_train_small, 'LineWidth', 2);
plot(1:hyperparameters.num_iterations, cross_entropy_valid_small, 'LineWidth', 2);
lgd = legend('train\_small', 'valid', 'Location', 'NorthEast');
set(lgd, 'FontSize', 12);
xlabel('Iteration', 'FontSize', 12);
ylabel('Cross entropy', 'FontSize', 12);

clear t lgd;
