function [model] = svmAvg(X,y,lambda,maxIter)
% SVMAVG minimizes the SVM objective function by stochastic method based on
% the running average of w.
%
% Yuanbo Han, 2017-11-18.

% Add the bias variable.
[n,d] = size(X);
X = [ones(n,1), X];

% Use the transpose to accelerate the program.
Xt = X';

% Initialize the values of regression parameters.
w = zeros(d+1,1);

% The running average of w
w_bar = w;

% Apply stochastic method based on the running average of w.
for t = 1:maxIter
    if mod(t-1,n) == 0
        % Plot our progress.
        % (turn this off for acceleration)
        
        objValues(1+(t-1)/n) = (1/n)*sum(max(0,1-y.*(X*w_bar))) + (lambda/2)*(w_bar'*w_bar);
        semilogy([0:t/n],objValues);
        pause(.1);
    end
    
    % Pick a random training example.
    i = randi(n);
    
    % Compute the i-th subgradient.
    [~, sg] = hingeLossSubGrad(w,Xt,y,i);
    
    % Set step size.
    alpha = 1 / (lambda * t);
    
    % Take stochastic subgradient step.
    w = w - alpha * (sg + lambda * w);
    
    % Renew the running average of w.
    w_bar = (t-1)/t * w_bar + 1/t * w;
end

model.w = w_bar;
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
