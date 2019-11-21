function [ce, frac_correct] = evaluate(targets, y)
% Compute evaluation metrics.
% Inputs:
%   targets : N x 1 vector of binary targets. Values should be either 0 or 1.
%   y       : N x 1 vector of probabilities.
% Outputs:
%   ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we
%                  want to compute CE(targets, y).
%   frac_correct : (scalar) Fraction of inputs classified correctly.
%
% Yuanbo Han, 2017-11-12.

ce = mean( - targets .* log(y) - (1-targets) .* log(1-y) );
frac_correct = ( sum(targets==1 & y>=0.5) + sum(targets==0 & y<0.5) ) / size(y,1);
end
