function [Indices] = cvInd(n,K)
%CVIND(K) returns a K-by-(n/K) matrix (take floor(n/K) instead when n/K is
% not an integer) of randomly generated indices for a K-fold
% cross-validation of n-time observations, where row vectors are K disjoint
% subsets of [1:n] with the same length. K defaults to 5 when omitted.
%
% Yuanbo Han, Oct. 8, 2017

if ~exist('K','var')
    K = 5;
end

step = floor(n/K);
R_ind = randperm(n);
Indices = zeros(K,step);

for i = 1:K
    strtindex = (i-1) * step + 1;
    stpindex = strtindex + step - 1;
    Indices(i,:) = R_ind(strtindex:stpindex);
end
end
