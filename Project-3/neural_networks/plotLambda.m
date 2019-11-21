% Edited by Yuanbo Han, Dec. 8, 2017.

load digits.mat;
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = d + 1;

% Apply the same transformation to the validation data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];

% Choose network structure
nHidden = [120];

% Count number of parameters
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;

maxIter = 100000;
stepSize = 1e-3;
funObj = @(w,i,lambda)MLP_L2(w, X(i,:), yExpanded(i,:), nHidden, ...
    nLabels, lambda);

lambda = zeros(1,10);
lambda(1) = 0.001;
for i = 2:10
    lambda(i) = lambda(i-1) * 2;
end

validError = zeros(1,length(lambda));
for l = 1:length(lambda)
    tic
    for k = 1:10
        % Initialize weights 'w'
        w = randn(nParams,1);
        
        % Train with stochastic gradient
        for iter = 1:maxIter
            i = ceil(rand*n);
            [~,g] = funObj(w,i,lambda(l));
            w = w - stepSize*g;
        end
        
        % Evaluate validation error
        yhat = MLPclassificationPredict(w, Xvalid, nHidden, nLabels);
        validError(l) = validError(l) + 1/10 * sum(yhat~=yvalid)/t;
    end
    fprintf('lambda = %.3f\n', lambda(l));
    fprintf('Average validation error = %f\n', validError(l));
    toc
end

figure;
semilogx(lambda, validError);
set(gca, 'XTick', lambda);
xlabel('\lambda', 'FontSize', 12);
ylabel('Validation Error', 'FontSize', 12);
title('L2-Regularization', 'FontSize', 14);
