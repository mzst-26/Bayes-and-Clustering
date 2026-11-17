function [matched, clusterToClass] = matchLabels(predicted, y)

%cluster numbers are random (cluster 1 might be true class 3),
%so I name each cluster after the true class that most of its points belong to
%majority vote with mode. 
%clusterToClass(k)=the class given to cluster k

clusterToClass = zeros(1, max(predicted));

matched = zeros(size(predicted));

for k = 1:max(predicted)

    if any(predicted == k)

        clusterToClass(k) = mode(y(predicted == k));
        matched(predicted == k) = clusterToClass(k);

    end
end


%if two clusters get the same class the accuracy would be unfair
if numel(unique(matched)) < numel(unique(predicted))

    disp('warning two clusters were given the same class');
end

end
