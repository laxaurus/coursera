function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_set = [0.01, 0.03, 0.1, 0.3, 1,3, 10, 30];
sigma_set = C_set;
n = columns(C_set);

% pred_error is a matrix of (n*n or 64) rows  X 5 columns to store the results
% where each row contains
% [row#i, col#j, pred_error, C@i th row, sigma@j th row]
%
pred_error = zeros(n * n, 5);
k = 1;

%
% 'x1' and 'x2' are dummy parameters. 
% They are filled-in at runtime when svmTrain() calls your kernel function.
% taken out from Discussion Forums | Week 7 | FAQ for Week 7 and programming exercise 6
%
x1 = x2 = 0;


for i = 1:n
  for j = 1:n
    
    C = C_set(:,i);
    sigma = sigma_set(:, j);
    fprintf ("run number: C[%d]=%f sigma[%d]=%f ", i, C, j, sigma);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predictions = svmPredict(model, Xval);  
    error = mean(double(predictions ~= yval));
    fprintf (" error = %f\n", error);
    pred_error(k,:) = [i j error C sigma];
    k=k+1;
  endfor
endfor
[v, i] = min(pred_error(:,3));
%pred_error(i,:)
C = pred_error(i, 4);
sigma = pred_error(i, 5);
fprintf ("------------------------------------");
fprintf ("min cost is %f when C = %f and sigma = %f\n", pred_error(i,3), C, sigma);





% =========================================================================

end
