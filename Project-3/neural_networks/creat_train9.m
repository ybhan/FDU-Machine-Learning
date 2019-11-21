% Edited by Yuanbo Han, Dec. 9, 2017.

load digits.mat;

% Artificially creat more training examples.
tic
[n,d] = size(X);
Xright = zeros([n,d]);
Xleft = zeros([n,d]);
Xdown = zeros([n,d]);
Xup = zeros([n,d]);
Xclock = zeros([n,d]);
Xanticlock = zeros([n,d]);
Xbig = zeros([n,d]);
Xsmall = zeros([n,d]);
for i = 1:size(X,1)
    Xfig = reshape(X(i,:),16,16);
    % Translations
    temp = imtranslate(Xfig, [0,1]);
    Xright(i,:) = temp(:);
    temp = imtranslate(Xfig, [0,-1]);
    Xleft(i,:) = temp(:);
    temp = imtranslate(Xfig, [1,0]);
    Xdown(i,:) = temp(:);
    temp = imtranslate(Xfig, [-1,0]);
    Xup(i,:) = temp(:);
    % Rotations
    temp = imrotate(Xfig, 5, 'crop');
    Xclock(i,:) = temp(:);
    temp = imrotate(Xfig, -5, 'crop');
    Xanticlock(i,:) = temp(:);
    % Resizing
    temp = imresize(Xfig, 1.1, 'OutputSize', [16,16]);
    Xbig(i,:) = temp(:);
    temp = imresize(Xfig, 0.9, 'OutputSize', [16,16]);
    Xsmall(i,:) = temp(:);
end

X = [X; Xright; Xleft; Xdown; Xup; Xclock; Xanticlock; Xbig; Xsmall];
y = repmat(y,9,1);
toc

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

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
w = randn(nParams,1);

% Train with stochastic gradient
maxIter = 10000;
stepSize = 1e-3;% * 3;
%momentumStrength = 0.9;
%delta = 0;
%lambda = 0.03;
funObj = @(w,i)MLPclassificationLoss_mat(w,X(i,:), ...
    yExpanded(i,:), nHidden, nLabels);

tic
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = MLPclassificationPredict(w,Xvalid,nHidden,nLabels);
        fprintf('Training iteration = %d, validation error = %f\n', ...
            iter-1,sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand*n);
    [~,g] = funObj(w,i);
    %    delta = stepSize * g - momentumStrength * delta;
    %    w = w - delta;
    w = w - stepSize * g;
end

% Evaluate test error
yhat = MLPclassificationPredict(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
toc
