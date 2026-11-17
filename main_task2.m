%COMP3003 Task 2
%k-means mykm.m is built from first principle
%for GMM I used MATLAB fitgmdist fitGMM.m

clear; 
close all; 
clc;

%my student number used as the random seed
studentSeed =10869518;
%the K values I am testing
kValues =2:8;

%check if figures directory exist if not make one. 
if ~exist('figures', 'dir')
    mkdir('figures');
end

%check if results text file exist if not make one. 
if exist('results_log.txt', 'file')
    delete('results_log.txt');
end

%save everything printed to a text file
diary('results_log.txt');


%Make the dataset

%5 clusters with 2 dimensions
%different sizes
%C1 is big, C2 is small and tight, C3 is long in x,
%C4 is long in y and C5 overlaps a bit with C3

%                 mean x  mean y  sd x  sd y   points
clusterSettings = [  0.0     0.0     1.0   1.0   1500;
                     5.0     0.5     0.5   0.5    500;
                     3.0     5.5     2.0   0.6   1200;
                    -4.0     4.0     0.6   1.6    800;
                     5.5     8.0     1.0   1.0   1000 ];

rng(studentSeed, 'twister');
[X, y] = generateData(clusterSettings);
numPointsTotal = size(X,1);

%check the sample(mean, sd and correlation of each cluster)
disp('cluster   sample mean        sample sd        corr');
for k = 1:5
    pts = X(y == k,:);
    r = corrcoef(pts);
    fprintf('C%d   (%6.2f, %5.2f)   (%4.2f, %4.2f)   %+.3f\n', k, mean(pts), std(pts), r(1,2));
end

figure;
gscatter(X(:,1), X(:,2), y, lines(5), '.', 8);
axis equal;
title('Generated data (true clusters)');
print(gcf, 'figures/fig1_data.png', '-dpng', '-r200');


%my k-means with K = 5

rng(studentSeed + 1, 'twister');
%10 restarts with k-means++ starts
kmeansResult = mykm(X, 5, 10);

fprintf('\nK-means K=5: SSE = %.2f after %d iterations\n', kmeansResult.sse, kmeansResult.iters);
fprintf('SSE of the 10 restarts: %s\n', sprintf('%.2f ', kmeansResult.allSSE));
fprintf('points that changed cluster per iteration: %s\n', sprintf('%d ', kmeansResult.movedHistory));

%does the starting point matter? 20 single runs with each way of starting
rng(studentSeed + 4, 'twister');
ssePlus = zeros(20,1);
sseRandom = zeros(20,1);

for r = 1:20
    runPlus = mykm(X, 5, 1, 'plus');
    runRandom = mykm(X, 5, 1, 'random');
    ssePlus(r) = runPlus.sse;
    sseRandom(r) = runRandom.sse;
end

fprintf('single runs that reach the best SSE: k-means++ %d/20, random start %d/20\n', ...
    sum(ssePlus < kmeansResult.sse + 0.01), sum(sseRandom < kmeansResult.sse + 0.01));
fprintf('worst single run: k-means++ SSE = %.2f, random start SSE = %.2f\n', max(ssePlus), max(sseRandom));

[kmeansLabelsMatched, kmeansMap] = matchLabels(kmeansResult.labels, y);

figure('Position', [100 100 1200 360]);
subplot(1,3,1);

gscatter(X(:,1), X(:,2), kmeansLabelsMatched, lines(5), '.', 6);
hold on;
plot(kmeansResult.C(:,1), kmeansResult.C(:,2), 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y');
axis equal; legend off;
title('K-means, K = 5');
subplot(1,3,2);
plot(kmeansResult.sseHistory, '-o');
xlabel('iteration'); ylabel('SSE (J)');
title('J goes down every iteration');
subplot(1,3,3);
plot(2:kmeansResult.iters, kmeansResult.movedHistory(2:end), '-o');
xlabel('iteration'); ylabel('points that changed cluster');
title('Reassignments');
print(gcf, 'figures/fig2_kmeans_k5.png', '-dpng', '-r200');


%Choosing K for kmeans try K = 2 to 8
rng(studentSeed + 3, 'twister');
kmeansSSEByK = zeros(1, length(kValues));

%one column per K
kmeansLabelsByK = zeros(numPointsTotal, length(kValues));

for kIndex = 1:length(kValues)

    kmeansRun = mykm(X, kValues(kIndex), 10);
    kmeansSSEByK(kIndex) = kmeansRun.sse;
    kmeansLabelsByK(:,kIndex) = kmeansRun.labels;

end

%elbow, how much SSE drops at each step in percent
sseDrop = 100 * -diff(kmeansSSEByK) ./ kmeansSSEByK(1:end-1);

%two indices from evalclusters
kmSil = evalclusters(X, kmeansLabelsByK, 'silhouette', 'Distance', 'Euclidean');
kmCH = evalclusters(X, kmeansLabelsByK, 'CalinskiHarabasz');



fprintf('\n K   SSE       drop to next K   silhouette   CH\n');
for kIndex = 1:length(kValues)
    if kIndex < length(kValues)
        dropText = sprintf('%5.1f%%', sseDrop(kIndex));
    else
        dropText = '   -  ';
    end
    fprintf(' %d   %8.1f   %s          %.4f       %7.1f\n', kValues(kIndex), kmeansSSEByK(kIndex), ...
        dropText, kmSil.CriterionValues(kIndex), kmCH.CriterionValues(kIndex));
end
fprintf('best K: silhouette = %d, CH = %d (elbow: see the SSE drops)\n', kmSil.OptimalK, kmCH.OptimalK);


%why do some pick K = 4?
%looking at where the true clusters go when k is 4
disp('K = 4 k-means (rows: true C1..C5, columns: clusters):');
disp(crosstab(y, kmeansLabelsByK(:, kValues == 4)));

figure('Position', [100 100 1200 330]);

subplot(1,3,1); plot(kValues, kmeansSSEByK, '-o'); title('SSE (elbow)'); xlabel('K');
subplot(1,3,2); plot(kValues, kmSil.CriterionValues, '-o'); title('Silhouette (higher better)'); xlabel('K');
subplot(1,3,3); plot(kValues, kmCH.CriterionValues, '-o'); title('Calinski-Harabasz (higher better)'); xlabel('K');
print(gcf, 'figures/fig3_kmeans_chooseK.png', '-dpng', '-r200');





%GMM with K=5 which is started from k means result
gmmModel = fitGMM(X, kmeansResult.labels, 5);
fprintf('\nGMM K=5 log-likelihood = %.3f\n', -gmmModel.NegativeLogLikelihood);

%hard answer: most likely Gaussian
gmmLabels = cluster(gmmModel, X);   

%soft answer: probability of each Gaussian
gmmPosterior = posterior(gmmModel, X);      
[gmmLabelsMatched, gmmMap] = matchLabels(gmmLabels, y);


%what the GMM learned for each component
disp('component   weight   mean            sd             corr');

for k = 1:5
    %the component named after true cluster k
    j = find(gmmMap == k, 1);   
    S = gmmModel.Sigma(:,:,j);

    fprintf('C%d          %.3f    (%5.2f, %5.2f)   (%4.2f, %4.2f)   %+.3f\n', k, gmmModel.ComponentProportion(j), gmmModel.mu(j,:), sqrt(S(1,1)), sqrt(S(2,2)), S(1,2) / sqrt(S(1,1)*S(2,2)));
end



%points the GMM is not sure about(no Gaussian above 0.9)
unsure = max(gmmPosterior, [], 2) < 0.9;

fprintf('unsure points %d (%.1f%%), per true cluster C1 to C5: %s\n', sum(unsure), 100*mean(unsure), sprintf('%d ', accumarray(y(unsure), 1, [5 1])));

% plot the fitted pdf
gridX = linspace(min(X(:,1)) - 1, max(X(:,1)) + 1, 250);
gridY = linspace(min(X(:,2)) - 1, max(X(:,2)) + 1, 250);
[gx, gy] = meshgrid(gridX, gridY);
gridPoints = [gx(:) gy(:)];
pdfGrid = reshape(pdf(gmmModel, gridPoints), size(gx));



figure;
plot(X(:,1), X(:,2), '.', 'Color', [0.75 0.75 0.75]);
hold on;
contour(gx, gy, pdfGrid, 12);
plot(X(unsure,1), X(unsure,2), 'r.');
plot(gmmModel.mu(:,1), gmmModel.mu(:,2), 'kx', 'MarkerSize', 12, 'LineWidth', 2);
axis equal;
title('GMM, K = 5: pdf contours, unsure points in red');
print(gcf, 'figures/fig4_gmm_k5.png', '-dpng', '-r200');




%what K is best for the GMM K=2 to 8

gmmBICByK = zeros(1, length(kValues));
gmmAICByK = zeros(1, length(kValues));
gmmLabelsByK = zeros(numPointsTotal, length(kValues));

for kIndex = 1:length(kValues)

    %start EM from the k-means result with the same K
    gmmFit = fitGMM(X, kmeansLabelsByK(:,kIndex), kValues(kIndex));

    gmmBICByK(kIndex) = gmmFit.BIC;
    gmmAICByK(kIndex) = gmmFit.AIC;

    %assign clusters and fix the numbering if any are empty
    [~, ~, gmmLabelsByK(:,kIndex)] = unique(cluster(gmmFit, X));   

end

gmmSil = evalclusters(X, gmmLabelsByK, 'silhouette', 'Distance', 'Euclidean');

[~, bestBIC] = min(gmmBICByK);
[~, bestAIC] = min(gmmAICByK);

fprintf('\n K   BIC        AIC        silhouette (GMM)\n');

for kIndex = 1:length(kValues)

    fprintf(' %d   %9.1f  %9.1f  %.4f\n', kValues(kIndex), gmmBICByK(kIndex), gmmAICByK(kIndex), ...
        gmmSil.CriterionValues(kIndex));

end

fprintf('best K: BIC = %d, AIC = %d, silhouette = %d\n', kValues(bestBIC), kValues(bestAIC), gmmSil.OptimalK);



figure('Position', [100 100 900 320]);
subplot(1,2,1);
plot(kValues, gmmBICByK, '-o', kValues, gmmAICByK, '-s');
legend('BIC', 'AIC'); title('GMM BIC and AIC (lower better)'); xlabel('K');
subplot(1,2,2);
plot(kValues, gmmSil.CriterionValues, '-o');
title('GMM silhouette (higher better)'); xlabel('K');
print(gcf, 'figures/fig5_gmm_chooseK.png', '-dpng', '-r200');



%compare k-means and GMM with the same metrics (K = 5)
kmMetrics = clusterMetrics(y, kmeansResult.labels, kmeansLabelsMatched);
gmMetrics = clusterMetrics(y, gmmLabels, gmmLabelsMatched);

kmSil5 = evalclusters(X, kmeansResult.labels, 'silhouette', 'Distance', 'Euclidean');
gmSil5 = evalclusters(X, gmmLabels, 'silhouette', 'Distance', 'Euclidean');

%best possible accuracy. label each point with the TRUE Gaussians and Bayes' rule
truePost = zeros(numPointsTotal, 5);

for k = 1:5
    truePost(:,k) =clusterSettings(k,5) *mvnpdf(X, clusterSettings(k,1:2), diag(clusterSettings(k,3:4).^2));
end

[~, bayesLabels] =max(truePost, [], 2);


fprintf('\nK = 5            K-means   GMM\n');
fprintf('accuracy         %.4f    %.4f   (best possible with the true Gaussians: %.4f)\n', kmMetrics.accuracy, gmMetrics.accuracy, mean(bayesLabels == y));
fprintf('wrong points     %d       %d\n', sum(kmeansLabelsMatched ~= y), sum(gmmLabelsMatched ~= y));
fprintf('purity           %.4f    %.4f\n', kmMetrics.purity, gmMetrics.purity);
fprintf('entropy          %.4f    %.4f   (lower is better)\n', kmMetrics.entropy, gmMetrics.entropy);
fprintf('homogeneity      %.4f    %.4f\n', kmMetrics.homogeneity, gmMetrics.homogeneity);
fprintf('completeness     %.4f    %.4f\n', kmMetrics.completeness, gmMetrics.completeness);
fprintf('V-measure        %.4f    %.4f\n', kmMetrics.vmeasure, gmMetrics.vmeasure);
fprintf('VI               %.4f    %.4f   (lower is better)\n', kmMetrics.VI, gmMetrics.VI);
fprintf('silhouette       %.4f    %.4f\n', kmSil5.CriterionValues, gmSil5.CriterionValues);
fprintf('recall C1..C5    k-means %s | GMM %s\n', sprintf('%.3f ', kmMetrics.recall), sprintf('%.3f ', gmMetrics.recall));
fprintf('precision C1..C5 k-means %s | GMM %s\n', sprintf('%.3f ', kmMetrics.precision), sprintf('%.3f ', gmMetrics.precision));
fprintf('parameters       %d        %d\n', 5*2, (5-1) + 5*2 + 5*3);


disp('Confusion matrices (rows: true C1..C5, columns: predicted):');
disp(kmMetrics.confusion);
disp(gmMetrics.confusion);

figure('Position', [100 100 1000 420]);
subplot(1,2,1);
confusionchart(y, kmeansLabelsMatched, 'Title', 'K-means, K = 5');
subplot(1,2,2);
confusionchart(y, gmmLabelsMatched, 'Title', 'GMM, K = 5');
print(gcf, 'figures/fig6_confusion.png', '-dpng', '-r200');

%decision boundaries
%nearest centroid
[~, kmGrid] = min(pdist2(gridPoints, kmeansResult.C), [], 2);   

%most likely Gaussian
gmGrid = cluster(gmmModel, gridPoints);                           

figure('Position', [100 100 1000 420]);
subplot(1,2,1);
imagesc(gridX, gridY, reshape(kmeansMap(kmGrid), size(gx)));
axis xy; hold on;
plot(X(:,1), X(:,2), 'k.', 'MarkerSize', 1);
plot(kmeansResult.C(:,1), kmeansResult.C(:,2), 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y');
axis equal tight; title('K-means regions (straight lines)');
subplot(1,2,2);
imagesc(gridX, gridY, reshape(gmmMap(gmGrid), size(gx)));
axis xy; hold on;
plot(X(:,1), X(:,2), 'k.', 'MarkerSize', 1);
axis equal tight; title('GMM regions (curved)');
colormap(lines(5));
print(gcf, 'figures/fig7_boundaries.png', '-dpng', '-r200');

diary off;
