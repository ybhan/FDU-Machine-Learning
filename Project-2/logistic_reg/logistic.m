function [f, df, y] = logistic(weights, data, targets, ~)
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
%	targets:    N x 1 vector of binary targets. Values should be either
%               0 or 1.
%   ~:          The hyperparameter structure is omitted.
%
% Outputs:
%	f:             The scalar error value (i.e. negative log likelihood).
%	df:            (M+1) x 1 vector of derivatives of error w.r.t. weights.
%   y:             N x 1 vector of probabilities.
%                  This is the output of the classifier.
%
% Yuanbo Han, 2017-11-12.

x = [ data, ones( size(data,1), 1 ) ];
z = x * weights;
y = sigmoid(z);
f = - z' * (targets - 1) - sum( log(y) );
df = - x' * ( targets - y );
end
