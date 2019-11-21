function [X_re] = PCAreconstruct(Y,V)
% PCARECONSTRUCT(Y,V) compose the corresponding projected vector of Y,
% where V is the eigenvectors used in PCA.
%
% Edited by Yuanbo Han, Dec. 25, 2017.
X_re = V * Y;
end
