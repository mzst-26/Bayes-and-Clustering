function out = mykm(X, numClusters, numRestarts, initMethod)

%k-means from first principle
%X=data (one point per row)
%numClusters= K, 
%numRestarts= how many times to run it
%initMethod= 'plus' for k-means++(default) or 'random' for K random data points
%k-means can get stuck depending on the starting points,
%so I run it a few times and keep the run with the smallest SSE

if nargin < 4
    initMethod = 'plus';
end

numPoints = size(X,1);
bestSSE = Inf;
allSSE = zeros(numRestarts,1);

for restart = 1:numRestarts

    %choose the starting centroids
    if strcmp(initMethod, 'random')

        %K different random points from the data
        [~, order] = sort(rand(numPoints,1));
        centroids = X(order(1:numClusters), :);

    else

        %k-means++, the first centre is a random point from the data
        centroids = X(ceil(rand*numPoints), :);

        %the other centres, points far from the centres already picked have a higher chance

        for k = 2:numClusters

            closestDistance = min(squaredDistances(X, centroids), [], 2);
            cumulativeDistance = cumsum(closestDistance);
            chosenPoint = find(cumulativeDistance >= rand*cumulativeDistance(end), 1);
            centroids(k,:) = X(chosenPoint,:);

        end

    end

    %assign and update until no point changes cluster
    clusterLabels = zeros(numPoints,1);
    %SSE (J) at every iteration
    sseHistory = [];
    %how many points changed cluster at every iteration
    movedHistory = [];

    for iteration = 1:300

        %assign, put each point into the cluster with the closest centre
        [minDistance, newLabels] = min(squaredDistances(X, centroids), [], 2);

        sseHistory(iteration) = sum(minDistance);
        movedHistory(iteration) = sum(newLabels ~= clusterLabels);

        %stop if no point changed cluster
        if movedHistory(iteration) == 0
            break
        end

        clusterLabels = newLabels;

        %update, move each centre to the mean of the points in its cluster
        for k = 1:numClusters
            if any(clusterLabels == k)
                centroids(k,:) = mean(X(clusterLabels == k,:), 1);
            end
        end

    end

    %keep this run if it is the best so far
    runSSE = sseHistory(end);
    allSSE(restart) = runSSE;

    if runSSE < bestSSE

        bestSSE = runSSE;
        out.labels = clusterLabels;
        out.C = centroids;
        out.iters = iteration;
        out.sseHistory = sseHistory;
        out.movedHistory = movedHistory;

    end

end

out.sse = bestSSE;
out.allSSE = allSSE;

end


function distances = squaredDistances(X, centroids)

%squared distance between every point and each centre
%squared is enough because we only need to compare them

distances = zeros(size(X,1), size(centroids,1));

for k = 1:size(centroids,1)
    distances(:,k) = sum((X - centroids(k,:)).^2, 2);
end

end
