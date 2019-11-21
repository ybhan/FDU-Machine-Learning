function [theta] = Ridge(X,y,d2)
%RIDGE(X,y,d2) returns a vector theta of coefficient estimates for a
% multilinear ridge regression of the responses in y on the predictors in
% X, where d2 is the regularization parameter. Here, X and y must have been
% standardized.
%
% Yuanbo Han, Oct. 7, 2017

theta = ( X'*X + d2.*eye(size(X,2)) )\X'*y ;
end
