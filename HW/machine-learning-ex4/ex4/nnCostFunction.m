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

% Part 1:
X_1 = [ones(m, 1), X];

z_2 = X_1 * Theta1';
X_2 = [ones(size(z_2, 1), 1), 1 ./ (1 + exp(-z_2))];

z_3 = X_2 * Theta2';
h = 1 ./ (1 + exp(-z_3));
J = 0;

for i = 1 : m
	yi = (zeros(num_labels, 1));
	yi(y(i)) = 1;
	J = J + (-log(h(i, :)) * yi - log(1 - h(i, :)) * (1 - yi));
end

J = J / m + lambda / (2*m) * ...
	(trace(Theta1(:, 2:end) * Theta1(:, 2:end)') + ...
	 trace(Theta2(:, 2:end) * Theta2(:, 2:end)'));


% Part 2:

for i = 1 : m

	% forward propagation
	a1 = [1, X(i, :)]; 					% a1: (1,401)

	z2 = a1 * Theta1';					% z2: (1,401) * (401, 25) = (1, 25)
	a2 = [1, 1 ./ (1 + exp(-z2))];		% a2: (1,26)

	z3 = a2 * Theta2';					% z3: (1,26) * (26 * 10) = (1, 10)
	a3 = 1 ./ (1 + exp(-z3));			% a3: (1,10)

	% calculate error vector
	yi = ((1:num_labels)' == y(i));		% yi: (10,1)

	delta3 = a3' - yi;					% delta3: (10,1)
	delta2 = (Theta2(:, 2:end)' * delta3) .* sigmoidGradient(z2');	
										% delta2: (25, 10) * (10,1) .* (25,1) = (25,1)

	Theta1_grad = Theta1_grad + delta2 * a1;
										%Theta1_grad: (25,1)*(1,401)=(25,401)
	Theta2_grad = Theta2_grad + delta3 * a2;
										%Theta2_grad: (10,1)*(1,26)=(10,26)
end

Theta1_grad = 1/m * Theta1_grad;
Theta2_grad = 1/m * Theta2_grad;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
