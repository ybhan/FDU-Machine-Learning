% Edited by Yuanbo Han, Dec. 9, 2017.
% Reference: http://blog.csdn.net/u010540396/article/details/52895074

load digits.mat;
n = size(X,1);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and reshape X to be an array of n pixels.
[X,mu,sigma] = standardizeCols(X);
X = reshape(X',16,16,n);

% Apply the same transformation to the validation/test data.
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = reshape(Xvalid',16,16,t);
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = reshape(Xtest',16,16,t2);

% The number of neurons
nConv = 20;
nHidden = 200;

% Initialize bias.
bias_c = randn(1, nConv);
bias_f = randn(1, nHidden);
% Initialize convolution kernels.
kernel_c = randn(5,5,nConv);
kernel_f = randn(12,12,nHidden);
% Initialize weights for the full-connecting layer.
weight_f = randn(nConv, nHidden);
weight_output = randn(nHidden, nLabels);

% Train with stochastic gradient.
maxIter = 100000;
stepSize = 1e-3;
tic;
for iter = 1:maxIter
    if mod(iter-1, round(maxIter/10)) == 0
        yhat = CNN_predict(Xvalid, kernel_c, kernel_f, weight_f, ...
            weight_output, bias_c, bias_f);
        fprintf('Training iteration = %d, validation error = %f\n', ...
            iter-1, sum(yhat~=yvalid)/t);
        toc;
        tic;
    end
    
    i = ceil(rand*n);
    train_data = X(:,:,i);
    
    % Convolution layer
    state_c = zeros(12,12,nConv);
    for k = 1:nConv
        state_c(:,:,k)=conv2(train_data,rot90(kernel_c(:,:,k),2),'valid');
        % apply tanh
        state_c(:,:,k) = tanh(state_c(:,:,k) + bias_c(1,k));
    end
    
    % Full-connected layer
    [state_f_pre,state_f_temp] = convolution_f(state_c,kernel_f,weight_f);
    % apply tanh
    state_f = zeros(1, nHidden);
    for h = 1:nHidden
        state_f(1,h) = tanh(state_f_pre(:,:,h) + bias_f(1,h));
    end
    
    % Output layer (Softmax)
    output = zeros(1, nLabels);
    for h = 1:nLabels
        output(1,h) = exp( state_f*weight_output(:,h) ) / ...
            sum( exp(state_f*weight_output) );
    end
    
    % Update weights, kernels and bias.
    [kernel_c, kernel_f, weight_f, weight_output, bias_c, bias_f] = ...
        CNN_update(stepSize, y(i), train_data, state_c, state_f, ...
        state_f_temp, output, kernel_c, kernel_f, weight_f, ...
        weight_output, bias_c, bias_f);
end

yhat = CNN_predict(Xtest, kernel_c, kernel_f, weight_f, weight_output, ...
    bias_c, bias_f);
fprintf('Test error with final model = %f\n', sum(yhat~=ytest)/t2);
toc;
