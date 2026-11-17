%Computes P(U=1 | x,y,z) with a naive Bayes and a joint (full) Bayes
%classifier for all 8 input combinations of Table 3, and tests whether
%x, y, z are conditionally independent given U.

clear; clc;

%xyzU
D = [  1  0  0  0;
       0  1  1  0;
       0  0  1  0;
       1  0  0  1;
       0  0  1  1;
       0  1  0  1;
       1  1  0  1 ];

F = D(:, 1:3);%Inputs are all rows column 1 to 3
U = D(:, 4);%outputs are all rows column 4 only=U

classes = [0 1];

%priors P(U) and naive Bayes likelihoods P(feature=1|U)
prior = zeros(1, 2);%create a one by two matrix
p1 = zeros(3, 2);% p1 is a 3 by 2 matrix p1(j,c)=P(f_j=1|U=classes(c))

%loop runs twice with one time value being one and ther other two
for c = 1:2
    rows = (U == classes(c));%For each class 1s and 0s
    prior(c) = mean(rows);%mean(rows)=count÷total.
    p1(:, c) = mean(F(rows, :), 1)';%(x=0, U=0), (x=1, u=1).
end


%print the prior probabilities 
fprintf('P(U=0) = %.4f, P(U=1) = %.4f\n', prior);
fprintf('P(x=1|U), P(y=1|U), P(z=1|U) for U=0: %s\n', sprintf('%.4f ', p1(:,1)));
fprintf('P(x=1|U), P(y=1|U), P(z=1|U) for U=1: %s\n\n', sprintf('%.4f ', p1(:,2)));

%print the header
fprintf(' x y z | NB score U=0  NB score U=1 | NB P(U=1|.) | joint count  joint P(U=1|.)\n');

%Generate all 8 possible binary combinations for 3 variables.
combos = dec2bin(0:7) - '0';

%for each row
for i = 1:8
    q = combos(i, :);%the input we are scoring, e.g. [0 1 1]

    %naive Bayes
    score = zeros(1, 2);%score(1) is for U=0 amd score(2) is for U=1

    for c = 1:2
        lik = p1(:, c)';%P(x=1|U), P(y=1|U), P(z=1|U)
        lik(q == 0) = 1 - lik(q == 0);%if an input is 0, use 1 - P(input=1|U) instead
        score(c) = prior(c) * prod(lik);%prior times the three 
    end
    
    %divided by the total (Bayes rule) this is the evidence.
    pNB = score(2) / sum(score);

    %joint classifier
    match = all(F == q, 2);

    if any(match)
        matchingU = U(match);%classes of the rows that match q
        nMatch = sum(match);%how many rows match q
        nClass1 = sum(matchingU == 1);%how many of those are class 1
        pJoint = nClass1 / nMatch;%joint P(U=1 | q)
        jointTxt = sprintf('%d of %d - %.4f', nClass1, nMatch, pJoint);   % '1 of 2 -> 0.5000'

    else
        jointTxt = '0 of 0 - undefined (0/0)';%input never seen in the table
    end

    fprintf(' %d %d %d |   %.5f      %.5f   |   %.4f    | %s\n', q, score, pNB, jointTxt);
end

