% Compute and display the training and test errors for each computed value
% of theta.
% Edited by Yuanbo Han, Oct. 8, 2017

% Load the standardized data from 'stdData.mat'
load stdData;

% Compute the training and test errors for each computed value of theta
trainError = zeros(length(d2),1);
testError = zeros(length(d2),1);
for i = 1:length(d2)
    trainError(i) = sqrt( (y_train - X_train * theta(:,i))' * (y_train - X_train * theta(:,i) ) / ((y_train+y_train_bar)'*(y_train+y_train_bar)) );
    testError(i) = sqrt( (y_test - X_test * theta(:,i))' * (y_test - X_test * theta(:,i) ) / ((y_test+y_train_bar)'*(y_test+y_train_bar)) );
end

fprintf('trainError =\n\n');
disp(trainError);
fprintf('testError =\n\n');
disp(testError);

% Plot the training and test errors as a function of d2
figure;
semilogx(d2,trainError,d2,testError,'LineWidth',2);
xlabel('\delta^2','FontSize',15);
ylabel('\mid\midy-X\theta\mid\mid_2/\mid\midy\mid\mid_2','FontSize',15);
legend({'Train','Test'},'Location','NorthEast','FontSize',12);
grid on;

clear i;
