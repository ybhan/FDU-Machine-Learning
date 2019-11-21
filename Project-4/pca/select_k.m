% Edited by Yuanbo Han, Dec. 22, 2017.

load freyface.mat
X = double(X);  % convert to double

% Compute like SVD
N = size(X, 2);
[Vun, Dun] = eig(X*X'/N);
[lambda_un, order] = sort(diag(Dun));
Vun = Vun(:, order);
Xctr = X - repmat(mean(X, 2), 1, N);
[Vctr, Dctr] = eig(Xctr*Xctr'/N);
[lambda_ctr, order] = sort(diag(Dctr));
Vctr = Vctr(:, order);

% Eigen spectra (plot lambda_ctr)
figure;
plot(lambda_ctr, 'LineWidth', 1.5);
ylabel('\lambda', 'FontSize', 12);
title('Eigen spectra', 'FontSize', 15);

% Select k based on retention rate
original = sum(lambda_ctr);
new = 0;
r = 0.95;  % retention rate
for k=1:560
    new = new + lambda_ctr(end-k+1);
    if new / original >= r
        break;
    end
end
fprintf('k = %d, at the retention rate of %.2f%%\n', k, r*100);
