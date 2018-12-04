function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params


X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));

Theta_grad = zeros(size(Theta));
%fprintf('-----------size  num movies %d\n',  num_movies);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


% Add intercept term to X
X = [ones(num_movies, 1) X];
X_grad = zeros(size(X));
% Initialize fitting parameters
Theta = [zeros(num_users, 1) Theta];
Theta_grad = zeros(size(X));


regT = lambda / 2 * sum(sum(Theta(:,2:end).^2));
regX = lambda / 2 * sum(sum(X(:,2:end).^2));


% step 1 -> select only Rij == 1 -> X (5x 3+1) * Theta'(3+1 x4) -> Y (5x4)
% step 2 sum i of Y -> sum j of Y -> Y -> (1x1)
% refer to lecture notes pg 18
J = 1 / 2 * sum(sum( R .* (X * Theta' - Y).^2)) + regT + regX;




for i = 1:num_movies
  idx = find(R(i,:) == 1);
  %num_users_rated = size(idx);
  %dumpsize('idx', idx);
  %fprintf('there are %d users that rated the movie at %dth row (tot movies %d)\n', columns(num_users_rated), i, num_movies);
  
  Theta_temp = Theta(idx,:);
  %dumpsize('Theta_temp', Theta_temp);
  Y_temp = Y(i,idx);
  %dumpsize('Y_temp', Y_temp);
  X_grad(i,:) = (X(i,:) * Theta_temp' - Y_temp) * Theta_temp;
  

endfor


% for each user j, 
% -- determine the theta_gradient of user j
for j = 1:num_users
  % find all movies rated by user j
  % that means, select column j of R, go down through each row from 1 to num_movies, record the 
  % array index where the cell(i,j) == 1
  idx = find(R(:,j) == 1);
  %fprintf('there are %d movies that were rated by user %d (tot users %d)\n', columns(idx), i, num_users);
  % example:
  % R = [0;1;0;1;0]
  % then idx =  [2, 4, 5]
  % and Y at the jth column = [0;4;0;3;5]
  % thus Y_temp = [4;3;5]
  % X_temp row 1 = contents of row 2 and all columns from X
  % X_temp row 2 = contents of row 4 and all columns from X
  % X_temp row 3 = contents of row 5 and all columns from X
  Y_temp = Y(idx,j);
  X_temp = X(idx,:);
  Theta_grad(j,:) = (X_temp * Theta(j,:)' - Y_temp)' * X_temp;
  
endfor




% get rid of the intercept term X0 and theta0
X_grad = X_grad(:,2:end);
Theta_grad = Theta_grad(:,2:end);











% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
