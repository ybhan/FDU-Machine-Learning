% Edited by Yuanbo Han, Dec. 7, 2017.

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

maxIter = 10000;
stepSize = 1e-3;
validError = zeros(1,40);
% Choose network structure
for nHidden = 5:5:200
    tic
    % Count number of parameters
    nParams = d*nHidden(1);
    for h = 2:length(nHidden)
        nParams = nParams+nHidden(h-1)*nHidden(h);
    end
    nParams = nParams+nHidden(end)*nLabels;
    
    for k = 1:10
        % Initialize weights 'w'
        w = randn(nParams,1);
        
        % Train with stochastic gradient
        funObj = @(w,i)MLPclassificationLoss_mat(w, X(i,:), ...
            yExpanded(i,:), nHidden, nLabels);
        for iter = 1:maxIter
            i = ceil(rand*n);
            [~,g] = funObj(w,i);
            w = w - stepSize*g;
        end
        
        % Evaluate validation error
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        validError(nHidden/5) = validError(nHidden/5) + ...
            1/10 * sum(yhat~=yvalid)/t;
    end
    fprintf('nHidden = %d\n', nHidden);
    fprintf('Average validation error = %f\n', validError(nHidden/5));
    toc
end

figure;
plot(5:5:200, validError);
xlabel('nHidden', 'FontSize', 12);
ylabel('Validation Error', 'FontSize', 12);
title('Single Hidden Layer', 'FontSize', 14);
