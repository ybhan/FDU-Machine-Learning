function [model] = leastSquaresBias(X, y)
%LEASTSQUARESBIAS(X,y) Solves linear least-squares problem with a bias
% variable (assumes X'*X is invertible), where model.w(1) is the bias
% variable and model.w(2) the slope variable.
%
% Yuanbo Han, Oct. 7, 2017

X_1 = [ ones(size(X,1),1), X ];
w = (X_1'*X_1)\X_1'*y;

model.w = w;
model.predict = @predict;
end

function [yhat] = predict(model, Xhat)
w = model.w;
Xhat_1 = [ ones(size(Xhat,1),1), Xhat ];
yhat = Xhat_1*w;
end
