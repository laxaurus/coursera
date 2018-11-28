function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
m = rows(X);
for i = 1:m               % for each example
  min_xi = zeros(K,1);    
  for j = 1:K
    cj = centroids(j,:);    % extract the centroid at ith row
    % distance = sqrt(u1-v1)^2 + (u2-v2)^)
    % http://mathonline.wikidot.com/the-distance-between-two-vectors
    min_xi(j) = sum((X(i,:) - cj) .^ 2) ^(1/2);
    
  endfor
  %min_xi;
  [v, idx(i)] = min(min_xi);
endfor





% =============================================================

end

