function [log_prior, class_mean, class_var] = train_nb(train_data, train_label)
% TRAIN_NB trains a Naive Bayes binary classifier. All conditional
% distributions are Gaussian.
%
% Usage:
%   [log_prior, class_mean, class_var] = train_nb(train_data, train_label);
%
% Inputs:
%   train_data  : n_examples x n_dimensions matrix
%   train_label : n_examples x 1 binary label vector
%
% Outputs:
%   log_prior   : 2 x 1 vector, log_prior(i) = log(p(C=i)).
%   class_mean  : 2 x n_dimensions matrix, class_mean(i,:) is the mean
%                 vector for class i.
%   class_var   : 2 x n_dimensions matrix, class_var(i,:) is the variance
%                 vector for class i.
%
% Modified by Yuanbo Han, 2017-11-14: Omit the unused variable by '~', and
%                                     correct the comments.

SMALL_CONSTANT = 1e-10;

[~, n_dims] = size(train_data);
K = 2;

prior = zeros(K, 1);
class_mean = zeros(K, n_dims);
class_var = zeros(K, n_dims);

for k = 1 : K
    prior(k) = mean(train_label == (k-1));
    class_mean(k, :) = mean(train_data(train_label == (k-1), :), 1);
    class_var(k, :) = var(train_data(train_label == (k-1), :), 0, 1);
end

class_var = class_var + SMALL_CONSTANT;
log_prior = log(prior + SMALL_CONSTANT);

end
