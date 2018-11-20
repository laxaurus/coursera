function [error_test] = ...
    validateFinal(X_poly, y, X_poly_test, ytest, lambda)

m = rows(X_poly);
error_test = zeros(m, 1);

for i = 1:m
  theta = trainLinearReg(X_poly, y, lambda);  
  [error_test(i,:), grad] = linearRegCostFunction(X_poly_test, ytest, theta, 0);
  


  
  
endfor


error_test






% =========================================================================

end
