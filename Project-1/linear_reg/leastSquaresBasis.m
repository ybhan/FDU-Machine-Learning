function [model] = leastSquaresBasis(X,y,deg)
%LEASTSQUARESBASIS(X,y,deg) Solves least-squares problem with polynomial
% order deg (assumes X'*X is invertible), where model.w is the deg-by-1
% vector of coefficients.
%
% Yuanbo Han, Oct. 7, 2017

X_poly = ones(size(X,1), deg+1);
for i = 1:deg
    X_poly(:,i+1) = X.^i;
end

w = (X_poly'*X_poly)\X_poly'*y;

model.degree = deg;
model.w = w;
model.predict = @predict;
end

function [yhat] = predict(model,Xhat)
w = model.w;
deg = model.degree;

Xhat_poly = ones(size(Xhat,1), deg+1);
for i = 1:deg
    Xhat_poly(:,i+1) = Xhat.^i;
end

yhat = Xhat_poly*w;
end
