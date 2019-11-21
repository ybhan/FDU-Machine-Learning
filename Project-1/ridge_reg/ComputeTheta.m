% Compute theta for a range of d2 and plot them.
% Edited by Yuanbo Han, Oct. 8, 2017

% Load the standardized data in 'stdData.mat'
load stdData.mat;

% Compute the ridge regression solutions (theta) for a range of
% regularizers (d2)
d2 = logspace(-2,4,100);
theta = zeros(8,length(d2));
for i = 1:length(d2)
    theta(:,i) = Ridge(X_train,y_train,d2(i));
end

% Plot the regularization path (the values of each theta in the y-axis
% against d2 in the x-axis)
figure;
for i = 1:8
    semilogx( d2, theta(i,:), 'LineWidth', 2 );
    hold on;
end
xlabel('\delta^2','FontSize',15);
ylabel('\theta','FontSize',15);
legend(T(1:8),'Location','NorthEast','FontSize',12);
grid on;

clear i;
save stdData.mat d2 theta -append;
