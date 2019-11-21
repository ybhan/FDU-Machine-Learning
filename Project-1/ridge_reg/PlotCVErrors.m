% Compute the training and test errors for each d2 using cross-validation,
% and then plot the errors as a function of d2.
% Edited by Yuanbo Han, Oct. 8, 2017

% Load the standardized data from 'stdData.mat'
load stdData;

% Compute the training and test errors for each d2 using cross-validation
l = length(d2);
trainError = zeros(l,1);
testError = zeros(l,1);

% Below is a K-fold cross-validation.
K = 5;  % K can be changed to any integer in [2:n].

Indices = cvInd(n,K);
for i = 1:l
    [trainError(i),testError(i)] = cvErrors(X,y,d2(i),Indices);
end

% Plot the training and test errors as a function of d2
figure;
semilogx(d2,trainError,d2,testError,'LineWidth',2);
xlabel('\delta^2','FontSize',15);
ylabel('\mid\midy-X\theta\mid\mid_2/\mid\midy\mid\mid_2','FontSize',15);
legend({'Train','Test'},'Location','NorthEast','FontSize',12);
grid on;

% Find the d2 when testError takes minimum
[testError_min, index] = min(testError);
fprintf('When testError takes minimum, d2 = %.2f.\n', d2(index));

clear i l K;
