function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



%J = 1 / m * ((-1 .* y') * log(sigmoid(X * theta)) - (1 .- y)' * log(1 .- sigmoid(X * theta))) %...
    %+ lambda / (2 * m) * [0; (theta([2, rows(theta)],:) .^ 2)];
    
    
J = 1 / m * sum(((-1 .* y') * log(sigmoid(X * theta)) - (1 .- y)' * log(1 .- sigmoid(X * theta)))) ...    
    + lambda / (2 * m) * sum(theta([2: rows(theta)]) .^ 2);

%size(J)

%
% the dumb way
% first attempt:
% two step process: compute j0 without regularization and then j1-n with regularization
% 1. set up a matrix X1 of the same size as X. Next copy the row values from the first column of X into the matrix X1
% and fill the rest of the rows/columns with zeros. Apply the gradient formula, 
% Use X as the parameter in the sigmoid function 
% this result in an 1x1 matrix
% 2. set up a 2nd matrix X2N. It contains 0 in its first column, and data copied from col2 to col end of X
% apply the graident function
% this result in a 28x1 matrix
% finally,
% discard the first row of gradient X2N in Step 2 as it contains the regularized value
% return the gradient by adding the first result and X2N[2:end]
%
%
% 
X1 = [X(:, 1) zeros(m, columns(X) - 1)];
grad1 = 1 / m * X1' * (sigmoid(X * theta) - y);
%fprintf ('grad1: %f\n', grad1);


X2N = [[zeros(m, 1)] X(:, 2:end)];
%X2N
grad2N = 1 / m * X2N' * (sigmoid(X * theta) - y) + lambda / m * theta;
%grad2N
grad = [grad1(1, 1);  grad2N(2:end, 1)];
%fprintf ('------------> %f\n', grad(1:5));


%
%
% the quick way
% 
%

%grad =1/m * X' * (sigmoid(X* theta) - y) + lambda/m * theta .* [0; ones(length(theta)-1, 1)];
%fprintf ('------------> %f\n', grad(1:5));


% =============================================================

end
