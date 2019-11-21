function [trainError,testError] = cvErrors(X,y,d2,Indices)
%CVERRORS returns the trainError and the testError of a K-fold
% cross-validation on ridge regression, whose partition is given by Indices
% (an index matrix with K rows). X is the observation matrix, y the
% response vector and d2 the ridge parameter delta^2.
%
% Yuanbo Han, Oct. 8 ,2017

K = size(Indices,1);
trainErrorArray = zeros(K,1);
testErrorArray = zeros(K,1);

for i = 1:K
    X_test = X(Indices(i,:),:);
    y_test = y(Indices(i,:));
    temp = X;
    temp(Indices(i,:),:) = [];
    X_train = temp;
    temp = y;
    temp(Indices(i,:)) = [];
    y_train = temp;
    
    % Standardize the training data
    X_train_bar = mean(X_train);
    X_train_sigma = std(X_train,1);
    for j = 1:size(X_train,1)
        X_train(j,:) = (X_train(j,:) - X_train_bar) ./ X_train_sigma;
    end
    y_train_bar = mean(y_train);
    y_train = y_train - y_train_bar;
    
    % Standardize the test data
    for j = 1:size(X_test,1)
        X_test(j,:) = (X_test(j,:) - X_train_bar) ./ X_train_sigma;
    end
    y_test = y_test - y_train_bar;
    
    % Compute theta and errors
    theta = Ridge(X_train,y_train,d2);
    trainErrorArray(i) = sqrt( (y_train - X_train * theta)' * (y_train - X_train * theta) / ((y_train+y_train_bar)'*(y_train+y_train_bar)) );
    testErrorArray(i) = sqrt( (y_test - X_test * theta)' * (y_test - X_test * theta) / ((y_test+y_train_bar)'*(y_test+y_train_bar)) );
end

trainError = mean(trainErrorArray);
testError = mean(testErrorArray);
end
