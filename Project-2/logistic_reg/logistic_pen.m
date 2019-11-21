function [f, df, y] = logistic_pen(weights, data, targets, hyperparameters)
% Penalized logistic regression.
% Calculate log likelihood and derivatives with respect to weights.
%
% Note: N is the number of examples and
%       M is the number of features per example.
%
% Inputs:
% 	weights:    (M+1) x 1 vector of weights, where the last element
%               corresponds to bias (intercepts).
% 	data:       N x M data matrix where each row corresponds
%               to one data point.
%   targets:    N x 1 vector of targets class probabilities.
%   hyperparameters: The hyperparameter structure
%
% Outputs:
%	f:             The scalar error value (i.e. negative log liklihood
%                  + lambda/2 * weights(1:M)' * weights(1:M)).
%	df:            (M+1) x 1 vector of derivatives of error w.r.t. weights.
%   y:             N x 1 vector of probabilities.
%                  This is the output of the classifier.
%
% Yuanbo Han, 2017-11-13.

x = [ data, ones( size(data,1), 1 ) ];
z = x * weights;
y = sigmoid(z);
weights( length(weights) ) = 0;
f = - z' * (targets - 1) - sum( log(y) ) + hyperparameters.weight_regularization / 2 * weights' * weights;
df = - x' * ( targets - y ) + hyperparameters.weight_regularization * weights;
end
