% Project the data onto the top two eigenvectors,
% and plot the resulting 2D points.
Yctr = Vctr(:,end-1:end)' * X;
plot(Yctr(1,:), Yctr(2,:), '.');
explorefreymanifold(Yctr, X);

Yun = Vun(:,end-1:end)' * X;
plot(Yun(1,:), Yun(2,:), '.');
explorefreymanifold(Yun, X);
