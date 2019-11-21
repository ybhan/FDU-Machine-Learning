% Edited by Yuanbo Han, Dec. 26, 2017.
load digits.mat;

randConst = logspace(-3,2,20);
iters = 20;
repeat = 20;
logProbX_temp = zeros(iters,repeat);
logProbX = zeros(1,length(randConst));
for i = 1:length(randConst)
    fprintf('randConst = %f\n', randConst(i));
    for j = 1:repeat
        [~,~,~,logProbX_temp(:,j)] = ...
            mogEM_test(train3,2,iters,0.01,1,randConst(i));
        logProbX(i) = mean(logProbX_temp(end,:));
    end
    fprintf('logProbX = %f\n', logProbX(i));
end

figure;
semilogx(randConst, logProbX, 'LineWidth', 2);
xlabel('randConst', 'FontSize', 12);
ylabel('logP(X)', 'FontSize', 12);
