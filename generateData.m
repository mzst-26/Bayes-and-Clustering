function [X, y] = generateData(clusterSettings)

%makes the dataset
%5 clusters different sizes
%clusterSettings has one row per cluster[mean x, mean y, sd x, sd y, number of points]
%X=the points(one per row),y=the true cluster of each point(only used for checking)

X = [];
y = [];

for k = 1:size(clusterSettings,1)

    numPointsCluster = clusterSettings(k,5);
    mu = clusterSettings(k,1:2);

    %diagonal covariance matrix x and y are uncorrelated(no tilt)
    Sigma = diag(clusterSettings(k,3:4).^2);

    %standard normal numbers,norminv turns uniform rand numbers into normal ones
    %(inverse normal CDF), does the same job as randn in the lab
    normalSamples = norminv(rand(numPointsCluster, 2));

    %randn(m,2) * chol(Sigma)+ mean
    clusterPoints = normalSamples * chol(Sigma) + repmat(mu, numPointsCluster, 1);

    X = [X; clusterPoints];
    
    %save the real cluster number for checking later
    y = [y; k*ones(numPointsCluster,1)];

end

end
