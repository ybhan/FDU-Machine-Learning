function [model] = svm(X,y,lambda,maxIter)
% Add the bias variable.
[n,d] = size(X);
X = [ones(n,1), X];

% MATLAB indexes by columns. So if we are accessing rows, it will be faster
% to use the transpose.
Xt = X';

% Initialize the values of regression parameters.
w = zeros(d+1,1);

% Apply stochastic gradient method.
for t = 1:maxIter
    if mod(t-1,n) == 0
        % Plot our progress
        % (turn this off for speed)
        
        objValues(1+(t-1)/n) = (1/n)*sum(max(0,1-y.*(X*w))) + (lambda/2)*(w'*w);
        semilogy([0:t/n],objValues);
        pause(.1);
    end
    
    % Pick a random training example.
    i = randi(n);
    
    % Compute subgradient.
    [~, sg] = hingeLossSubGrad(w,Xt,y,i);
    
    % Set step size.
    alpha = 1 / (lambda * t);
    
    % Take stochastic subgradient step.
    w = w - alpha * ( sg + lambda * w );
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

function [loss,sg] = hingeLossSubGrad(w,Xt,y,i)
% Function value
wtx = w' * Xt(:,i);
loss = max(0, 1 - y(i) * wtx );

% Subgradient
if loss > 0
    sg = - y(i) * Xt(:,i);
else
    sg = zeros(size(Xt,1), 1);
end

end
