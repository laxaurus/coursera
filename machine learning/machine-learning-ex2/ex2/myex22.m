

clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('ex2data2.txt');

X = data(:, [1, 2]); y = data(:, 3);


X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters

% Compute and display initial cost and gradient for regularized logistic
% regression

%test_theta = ones(size(X,2),1);
%[cost, grad] = costFunctionReg(test_theta, X, y, 10);


a = randi(10,5,5)
b = randi(9,5,1)
a*b

[zeros(rows(a), 1) a(:, 2:end)];

c = [a(:,1) [zeros(rows(a), columns(a) -1)]] 
c * b

d = [zeros(rows(a), 1) [a(:,2:end)]]
d * b

e = (c*b) + (d*b) - (a*b)


