% Edited by Yuanbo Han, 2017-11-14.

% Clear workspace.
clear all;
close all;

% Load data.
load ../data/mnist_train;
load ../data/mnist_valid;

N = size(valid_inputs, 1);
K = 1:2:9;  % set of values of k
num = length(K);  % the number of values of k

% Compute the classification rates for each k.
classification_rate = zeros(num, 1);
for i = 1:num
    valid_labels = run_knn( K(i), train_inputs, train_targets, valid_inputs);
    classification_rate(i) = sum(valid_labels == valid_targets) / N;
end

% Plot the classification rate against k.
figure;
plot(K, classification_rate, 'LineWidth', 2);
title('kNN for Validation Set', 'FontSize', 15);
xlabel('k', 'FontSize', 12);
ylabel('Classification rate', 'FontSize', 12);
set(gca, 'XTick', K);
set(gca, 'XTickLabel', K);

clear i;
