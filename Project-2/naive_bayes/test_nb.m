function [prediction, accuracy] = test_nb(test_data, test_label, log_prior, class_mean, class_var)
% TEST_NB tests a learned Naive Bayes classifier.
%
% Usage:
%   [prediction, accuracy] = test_nb(test_data, test_label, log_prior, ...
%   class_mean, class_var);
%
% Inputs:
%   test_data  : n_examples x n_dimensions matrix
%   test_label : n_examples x 1 binary label vector
%   log_prior  : 2 x 1 vector, log_prior(i) = log(p(C=i)).
%   class_mean : 2 x n_dimensions matrix, class_mean(i,:) is the mean
%                vector for class i.
%   class_var  : 2 x n_dimensions matrix, class_var(i,:) is the variance
%                vector for class i.
%
% Outputs:
%   prediction : n_examples x 1 binary label vector
%   accuracy   : a real number
%
% Modified by Yuanbo Han, 2017-11-14: Correct the comments.

K = length(log_prior);
n_examples = size(test_data, 1);

log_prob = zeros(n_examples, K);

for k = 1 : K
    mean_mat = repmat(class_mean(k, :), [n_examples, 1]);
    var_mat = repmat(class_var(k, :), [n_examples, 1]);
    log_prob(:, k) = sum(-0.5 * (test_data - mean_mat).^2 ./ var_mat - 0.5 * log(var_mat), 2) + log_prior(k);
end

[~, prediction] = max(log_prob, [], 2);
prediction = prediction - 1;
accuracy = mean(prediction == test_label);

end
