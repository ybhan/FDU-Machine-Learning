function [model] = svmSAG(X,y,lambda,maxIter)
% SVMAVG minimizes the SVM objective function by stochastic average
% gradient (SAG) method.
%
% Yuanbo Han, 2017-11-18.

% Add the bias variable.
[n,d] = size(X);
X = [ones(n,1), X];

% Use the transpose to accelerate the program.
Xt = X';

% Initialize the values of regression parameters.
w = zeros(d+1,1);

% Compute the initial subgradients.
sg = zeros(d+1,n);
for j=1:n
    sg(:,j) = - y(j) * Xt(:,j);
end

% The average subgradient
m = mean(sg, 2);

% Apply stochastic average gradient (SAG) method.
for t = 1:maxIter
    if mod(t-1,n) == 0
        % Plot our progress.
        % (turn this off for acceleration)
        
        objValues(1+(t-1)/n) = (1/n)*sum(max(0,1-y.*(X*w))) + (lambda/2)*(w'*w);
        semilogy([0:t/n],objValues);
        pause(.1);
    end
    
    % Pick a random training example.
    i = randi(n);
    
    % Compute the i-th subgradient, and renew the average subgradient.
    m = m - sg(:,i) / n;
    [~, sg(:,i)] = hingeLossSubGrad(w,Xt,y,i);
    m = m + sg(:,i) / n;
    
    % Set step size.
    alpha = 1 / (lambda * t);
    
    % Take stochastic subgradient step.
    w = w - alpha * ( m + lambda * w );
end

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
d = size(Xhat,1);
Xhat = [ones(d,1), Xhat];
w = model.w;
yhat = sign(Xhat * w);
end

function [f,sg] = hingeLossSubGrad(w,Xt,y,i)
% Function value
wtx = w' * Xt(:,i);
f = max(0, 1 - y(i) * wtx );

% Subgradient
if f > 0
    sg = - y(i) * Xt(:,i);
else
    sg = zeros(size(Xt,1), 1);
end

end
