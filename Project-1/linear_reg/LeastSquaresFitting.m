% Edited by Yuanbo Han, Oct. 7, 2017

% Clear variables and close figures
clear all;
close all;

% Load data
load basisData.mat; % Loads X and y
[n,d] = size(X);

% Fit least-squares model for deg = 0 through deg = 10
for deg = 0:10
    fprintf('deg = %d\n', deg);
    model = leastSquaresBasis(X,y,deg);
    
    % Compute and report the training error
    yhat = model.predict(model,X);
    trainError = sum((yhat - y).^2)/n;
    fprintf('Training error = %.2f\n',trainError);
    
    % Compute and report the test error
    t = size(Xtest,1);
    yhat = model.predict(model,Xtest);
    testError = sum((yhat - ytest).^2)/t;
    fprintf('Test error = %.2f\n',testError);
    
    fprintf('\n');
end
