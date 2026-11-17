function m = clusterMetrics(y, predicted, matched)

%performance metrics for one clustering
%y=true classes
%predicted=cluster numbers
%matched = clusters named after classes

%clusters used as they are with no matching needed
%N(i,r)=points of class i in cluster r
N = crosstab(y, predicted);
n = sum(N(:));

%number of classes
q = size(N,1);

%n_r
clusterSize = sum(N, 1);

%purity, in each cluster count its biggest class
m.purity = sum(max(N, [], 1)) / n;

%entropy, how mixed each cluster is (0 = only one class), weighted by cluster size
P = N ./ clusterSize;

%0*log(0) counts as 0
P(P == 0) = 1;

clusterEntropy = -sum(P .* log(P), 1) / log(q);
m.entropy = sum(clusterSize / n .* clusterEntropy);

%homogeneity and completeness and V-measure use the entropies of classes (C) and clusters (K)
%entropy of a list of probabilitie
H = @(p) -sum(p(p > 0) .* log(p(p > 0)));    
pJoint = N / n;

%H(C)
HC = H(sum(pJoint, 2));

%H(K)
HK = H(sum(pJoint, 1));

%H(C|K)
HCgivenK = H(pJoint(:)) - HK;

%H(K|C)
HKgivenC = H(pJoint(:)) - HC;

%each cluster has only one class
m.homogeneity = 1 - HCgivenK / HC;

%each class is in only one cluster
m.completeness = 1 - HKgivenC / HK; 

m.vmeasure = 2 * m.homogeneity * m.completeness / (m.homogeneity + m.completeness);

%variation of information (lower is better)
m.VI = HCgivenK + HKgivenC;           

%need the matched labels
%rows=true class
%columns=predicted class
C = confusionmat(y, matched);   
m.accuracy = sum(diag(C)) / n;

%per class:TP/(TP + FN)
m.recall = (diag(C) ./ sum(C, 2))';    

%per class:TP/(TP + FP)
m.precision = (diag(C) ./ sum(C, 1)')';  
m.confusion = C;

end
