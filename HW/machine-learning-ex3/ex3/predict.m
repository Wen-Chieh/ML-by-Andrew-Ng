function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X_1 = [ones(m, 1), X]; 						% data of the first layer(input layer)
Z_2 = X_1 * Theta1';
X_2 = [ones(m, 1), 1 ./ (1 + exp(-Z_2))];	% data of the second layer(hidden layer)
Z_3 = X_2 * Theta2';
X_3 = 1 ./ (1 + exp(-Z_3));					% data of the last layer(output layer)
[v p] = max(X_3, [], 2);
p;






% =========================================================================


end
