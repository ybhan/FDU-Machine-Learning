% Edited by Yuanbo Han, 2017-11-13.

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
% Number of iterations
hyperparameters.num_iterations = 300;

[N, M] = size(train_inputs);
[N_small, M_small] = size(train_inputs_small);

penalty_parameters = logspace(-3,0,4);  % values of penalty parameter
num = length(penalty_parameters);  % the number of values of penalty parameter

rerun_times = 10;

cross_entropy_train = zeros(rerun_times, num);
cross_entropy_train_small = zeros(rerun_times, num);
cross_entropy_valid = zeros(rerun_times, num);
cross_entropy_valid_small = zeros(rerun_times, num);

%% Compute some stats for each penalty parameters.
for i = 1:num
    hyperparameters.weight_regularization = penalty_parameters(i);
    for r = 1:rerun_times
        fprintf('\n\nPENALTY PARAMETER = %.3f   RUN TIME = %d\n\n', hyperparameters.weight_regularization, r);
        
        % Randomly initialize the logistic regression weights.
        weights = randn( M+1, 1 );
        weights_small = randn( M_small+1, 1 );
        
        % Begin learning with gradient descent.
        for t = 1:hyperparameters.num_iterations
            
            % Find the error value and derivatives w.r.t. weights.
            [f, df, predictions] = logistic_pen(weights, ...
                train_inputs, ...
                train_targets, ...
                hyperparameters);
            
            [f_small, df_small, predictions_small] = logistic_pen(weights_small, ...
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
            [cross_entropy_train(r,i), frac_correct_train(r,i)] = evaluate(train_targets, predictions);
            [cross_entropy_train_small(r,i), frac_correct_train_small(r,i)] = evaluate(train_targets_small, predictions_small);
            
            % Update weights.
            weights = weights - hyperparameters.learning_rate .* df;
            weights_small = weights_small - hyperparameters.learning_rate .* df_small;
            
            % Find the cross entropy and fraction of correctly classified examples of validation data.
            predictions_valid = logistic_predict(weights, valid_inputs);
            predictions_valid_small = logistic_predict(weights_small, valid_inputs);
            
            [cross_entropy_valid(r,i), frac_correct_valid(r,i)] = evaluate(valid_targets, predictions_valid);
            [cross_entropy_valid_small(r,i), frac_correct_valid_small(r,i)] = evaluate(valid_targets, predictions_valid_small);
            
            % Print some stats.
            fprintf(1, 'ITERATION:%4i   NLOGL:%11.2f TRAIN_CE:%16.6f TRAIN_FRAC:%12.2f VALIC_CE:%16.6f VALID_FRAC:%12.2f\n',...
                t, f/N, cross_entropy_train(r,i), frac_correct_train(r,i)*100, cross_entropy_valid(r,i), frac_correct_valid(r,i)*100);
            fprintf('%17sNLOGL_SMALL:%5.2f TRAIN_SMALL_CE:%10.6f TRAIN_SMALL_FRAC:%6.2f VALID_CE_SMALL:%10.6f VALID_FRAC_SMALL:%6.2f\n',...
                '', f_small/N_small, cross_entropy_train_small(r,i), frac_correct_train_small(r,i)*100, cross_entropy_valid_small(r,i), frac_correct_valid_small(r,i)*100);
            
        end
    end
end

%% Plot the cross entropy and classification rate against penalty parameters.
figure;
subplot(2,2,1);
hold on;
title('mnist\_train', 'FontSize', 15);
plot(1:num, mean(cross_entropy_train, 1), 'LineWidth', 2);
plot(1:num, mean(cross_entropy_valid, 1), 'LineWidth', 2);
lgd = legend('train', 'valid', 'Location', 'NorthEast');
set(lgd, 'FontSize', 12);
xlabel('\lambda', 'FontSize', 12);
ylabel('Cross entropy', 'FontSize', 12);
set(gca, 'XTick', 1:num);
set(gca, 'XTickLabel', penalty_parameters);

subplot(2,2,2);
hold on;
title('mnist\_train', 'FontSize', 15);
plot(1:num, mean(frac_correct_train, 1), 'LineWidth', 2);
plot(1:num, mean(frac_correct_valid, 1), 'LineWidth', 2);
lgd = legend('train', 'valid', 'Location', 'NorthEast');
set(lgd, 'FontSize', 12);
xlabel('\lambda', 'FontSize', 12);
ylabel('Classification rate', 'FontSize', 12);
set(gca, 'XTick', 1:num);
set(gca, 'XTickLabel', penalty_parameters);

subplot(2,2,3);
hold on;
title('mnist\_train\_small', 'FontSize', 15);
plot(1:num, mean(cross_entropy_train_small, 1), 'LineWidth', 2);
plot(1:num, mean(cross_entropy_valid_small, 1), 'LineWidth', 2);
lgd = legend('train\_small', 'valid', 'Location', 'NorthEast');
set(lgd, 'FontSize', 12);
xlabel('\lambda', 'FontSize', 12);
ylabel('Cross entropy', 'FontSize', 12);
set(gca, 'XTick', 1:num);
set(gca, 'XTickLabel', penalty_parameters);

subplot(2,2,4);
hold on;
title('mnist\_train\_small', 'FontSize', 15);
plot(1:num, mean(frac_correct_train_small, 1), 'LineWidth', 2);
plot(1:num, mean(frac_correct_valid_small, 1), 'LineWidth', 2);
lgd = legend('train\_small', 'valid', 'Location', 'NorthEast');
set(lgd, 'FontSize', 12);
xlabel('\lambda', 'FontSize', 12);
ylabel('Classification rate', 'FontSize', 12);
set(gca, 'XTick', 1:num);
set(gca, 'XTickLabel', penalty_parameters);

clear i r t lgd;
