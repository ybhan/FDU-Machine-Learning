% Edited by Yuanbo Han, Dec. 8, 2017.

load digits.mat;
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = d + 1;

% Apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

% Choose network structure
nHidden = [120];

% Add a constant to each of the hidden layers
% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+(nHidden(h-1)+1)*nHidden(h);
end
nParams = nParams+(nHidden(end)+1)*nLabels;
w = randn(nParams,1);

% Train with stochastic gradient
maxIter = 10000;
stepSize = 1e-3; %* 3;
%momentumStrength = 0.9;
%delta = 0;
%lambda = 0.03;
funObj = @(w,i)MLP_addbias(w,X(i,:),yExpanded(i,:),nHidden,nLabels);

tic
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = MLP_addbias_predict(w,Xvalid,nHidden,nLabels);
        fprintf('Training iteration = %d, validation error = %f\n', ...
            iter-1, sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand*n);
    [~,g] = funObj(w,i);
    %    delta = stepSize * g - momentumStrength * delta;
    %    w = w - delta;
    w = w - stepSize * g;
end

% Evaluate test error
yhat = MLP_addbias_predict(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
toc
