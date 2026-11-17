function gmmModel = fitGMM(X, startLabels, numClusters)

%fits a GMM with EM using MATLAB fitgmdist
%EM is started from my k-means result instead of a random start
%each k-means cluster gives a starting mean and covariance and weight
%chose this method to have the same fit on every run


start.mu = zeros(numClusters, 2);
start.Sigma = zeros(2, 2, numClusters);
start.ComponentProportion = zeros(1, numClusters);


for k = 1:numClusters
    %mean for the current cluster.
    start.mu(k,:) = mean(X(startLabels == k,:), 1);

    %covariance matrix for the current cluster.
    start.Sigma(:,:,k) = cov(X(startLabels == k,:));

    %proportion of points in the current cluster
    start.ComponentProportion(k) = mean(startLabels == k);

end

%full covariance, each Gaussian can stretch in its own way
% a tiny ridge so the covariances stay invertible, and run the EM until it really stops

options = statset('MaxIter', 1000, 'TolFun', 1e-10);

gmmModel = fitgmdist(X, numClusters, 'Start', start, 'CovarianceType', 'full', ...
    'RegularizationValue', 1e-6, 'Options', options);

end
