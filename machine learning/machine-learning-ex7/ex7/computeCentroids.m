function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for k = 1:K
  % obtain a vector that contains only k in the idx vector
  idx_konly = (idx == k);
  % count the number of examples that are assigned centroid k
  num_ex_assigned_k = sum(idx_konly);
  % compute the mean value for centroid k
  % multiplying idx_konly with each element in X results in a matrix that have all the
  % unwanted values zeroed out, leaving the ones that are assigned k
  centroids(k,:) = 1 / abs(num_ex_assigned_k) * sum(idx_konly .* X);
endfor





% =============================================================


end

