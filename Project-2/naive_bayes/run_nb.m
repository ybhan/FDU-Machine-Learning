% Learn a Naive Bayes classifier on the digit dataset, evaluate its
% performance on training and test sets, then visualize the mean and
% variance for each class.
% Edited by Yuanbo Han, 2017-11-14.

% Clear workspace and close figures.
clear all;
close all;

% Load data.
load ../data/mnist_train;
load ../data/mnist_test;

% Add your code here (it should be less than 10 lines).
[log_prior, class_mean, class_var] = train_nb(train_inputs, train_targets);
[prediction_train, accuracy_train] = test_nb(train_inputs, train_targets, log_prior, class_mean, class_var);
[prediction_test, accuracy_test] = test_nb(test_inputs, test_targets, log_prior, class_mean, class_var);

fprintf('Training accuracy = %5.2f%%\n', accuracy_train * 100);
fprintf('Test accuracy = %9.2f%%\n', accuracy_test * 100);

plot_digits(class_mean);  % mean visualization
plot_digits(class_var);  % variance visualization
