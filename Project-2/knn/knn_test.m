% Edited by Yuanbo Han, 2017-11-14.

% Clear workspace.
clear all;
close all;

% Load data.
load ../data/mnist_train;
load ../data/mnist_test;

N = size(test_inputs, 1);
k = 5;  % my chosen value of k
K = [k-2, k, k+2];

% Compute the classification rates for each k.
classification_rate = zeros(3, 1);
for i = 1:3
    test_labels = run_knn( K(i), train_inputs, train_targets, test_inputs);
    classification_rate(i) = sum(test_labels == test_targets) / N;
end

% Plot the classification rate against k.
figure;
plot(K, classification_rate, 'LineWidth', 2);
title('kNN for Test Set', 'FontSize', 15);
xlabel('k', 'FontSize', 12);
ylabel('Classification rate', 'FontSize', 12);
set(gca, 'XTick', K);
set(gca, 'XTickLabel', K);

clear i;
