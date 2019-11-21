% Edited by Yuanbo Han, Dec. 25, 2017.

k = 80;
V = Vctr(:,end-k+1:end);

% Example for an original picture
figure;
showfreyface(X(:,50));  % the 50th sample in the data
figure;
showfreyface(PCAreconstruct(V'*(X(:,50)-mean(X,2)),V)+mean(X,2));

% Example for a random picture
R = randi([1,255], 560, 1);
figure;
showfreyface(R);
figure;
showfreyface(PCAreconstruct(V'*(R-mean(X,2)),V)+mean(X,2));

% Add noise for an original picture
X_noise = X(:,50) + 10*randn(560,1); % the 50th sample in the data
figure;
showfreyface(X_noise);
figure;
showfreyface(PCAreconstruct(V'*(X_noise-mean(X,2)),V)+mean(X,2));
