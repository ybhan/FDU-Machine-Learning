% Completed by Yuanbo Han, Dec. 26, 2017.
load digits;
x = [train2, train3];
%-------------------- Add your code here --------------------------------
% Train a MoG model with 20 components on all 600 training vectors
% with both original initialization and your kmeans initialization.

% Original initialization
[p1,mu1,vary1,logProbX1] = mogEM(x,20,10,0.01,0);
logProb1 = sum(mogLogProb(p1,mu1,vary1,x))

% K-means initialization
[p2,mu2,vary2,logProbX2] = mogEM_kmeans(x,20,10,0.01,0);
logProb2 = sum(mogLogProb(p2,mu2,vary2,x))
