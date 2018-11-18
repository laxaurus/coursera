function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% part 1

X1_1 = [ones(m, 1) X];
z2 = X1_1 * Theta1';
a2 = sigmoid(z2);
a2_1 = [ones(m, 1) a2];
z3 = a2_1 * Theta2';
a3 = sigmoid(z3);
%sel = randperm(size(a3, 1));
%sel = sel(1:20);
%out = a3(sel,:)


% This method uses an indexing trick to vectorize the creation of 'y_matrix', 
% where each element of 'y' is mapped to a single-value row vector copied from an eye matrix.
% check the notes in machine learning / resources /programming exercise 4 

Theta1_no_bias = Theta1(:, 2:end);
Theta2_no_bias = Theta2(:, 2:end);
%sum(sum(Theta1_no_bias .^ 2))
%sum(sum(Theta2_no_bias .^ 2))
J_reg = lambda / (2 * m) * ...
        (sum(sum(Theta1_no_bias .^ 2)) + sum(sum(Theta2_no_bias .^ 2)));


y_matrix = eye(num_labels)(y,:);
J = 1/m * sum(sum(-y_matrix .* log(a3) .- (1 .- y_matrix) .* log(1 - a3))) ...
    + J_reg;


% part 2
delta2 = zeros(num_labels, hidden_layer_size + 1);
delta1 = zeros(hidden_layer_size, input_layer_size + 1);
Theta2T = Theta2(:, 2:end)';
for t = 1: m
  X1_1t = X1_1(t,:);
  z2_t = X1_1t * Theta1';
  % the next line is commented out because a2_t has
  % already been computed above
  %a2_t = sigmoid(z2_t);
  
  
  a2_t = a2(t,:);
  a2_1t = [1 a2_t];
  % the next two lines are commented out because the values 
  % are available from previous computations
  %z3_t = a2_1t * Theta2';
  %a3_t = sigmoid(z3_t);
  
  a3_t = a3(t,:);

  d3_t = a3_t' - y_matrix(t,:)'; 
  % remove bias in Theta2 
  % refer to resources| programming ex.4 Step 7
  
  % the theta2 transpose is taken out of the loop
  % to prevent it from being computed over and over 
  %d2_t = Theta2(:, 2:end)' * d3_t .* sigmoidGradient(z2_t)';
  d2_t =  Theta2T * d3_t .* sigmoidGradient(z2_t)';
  
  delta2 = delta2 + d3_t * a2_1t;
  delta1 = delta1 + d2_t * X1_1t;
  
endfor



reg1 = lambda / m * [zeros(rows(Theta1_no_bias), 1)  Theta1_no_bias];
reg2 = lambda / m * [zeros(rows(Theta2_no_bias), 1)  Theta2_no_bias];
Theta1_grad = 1/m * delta1 + reg1;
Theta2_grad = 1/m * delta2 + reg2;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
