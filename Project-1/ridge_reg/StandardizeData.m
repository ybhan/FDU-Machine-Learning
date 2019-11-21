% Standardize the original data and save them into 'stdData.mat'.
% Edited by Yuanbo Han, Oct. 7, 2017

% Clear variables and close figures
clear all;
close all;

% Get the original data from 'prostate.data.txt'
file = importdata('prostate.data.txt');
X = file.data(:,1:8);
y = file.data(:,9);
T = file.colheaders;
n = size(X,1);

% Shuffle to obtain train and test data
R = randperm(n);
X_train = X(R(1:50),:);
X_test = X(R(51:n),:);
y_train = y(R(1:50));
y_test = y(R(51:n));

clear file R;

% Standardize the training data
X_train_bar = mean(X_train);
X_train_sigma = std(X_train,1);
for i = 1:50
    X_train(i,:) = (X_train(i,:) - X_train_bar) ./ X_train_sigma;
end
y_train_bar = mean(y_train);
y_train = y_train - y_train_bar;

% Standardize the test data
for i = 1:(n-50)
    X_test(i,:) = (X_test(i,:) - X_train_bar) ./ X_train_sigma;
end
y_test = y_test - y_train_bar;

clear i;
save stdData;
