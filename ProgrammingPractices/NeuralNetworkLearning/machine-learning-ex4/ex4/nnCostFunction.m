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


% --------------   Part 1   -------------- %





%Format y
Y = [1:num_labels] == y;
Y = Y';

D1 = 0;
D2 = 0;


%Add a column of '1's to the left side of X for bias input
a1 = [ones(rows(X), 1), X];

%For every training example...
for i = 1:rows(a1)

	%Grab the example
	ex = a1(i,:);
	
	%  Put it through the first hidden layer
	z2 = Theta1 .* ex;
	z2 = sum(z2, 2);
	a2 = [1; sigmoid(z2)];
	
	%  Put it through the second hidden layer
	z3 = Theta2 .* a2';
	a3 = sigmoid( sum(z3, 2) );
	
	
	%  Compute J
	A = log(a3);
	B = log(1 - a3);
	
	%Grab this examples respective Y
	Yk = Y(1:num_labels, i);
	
	temp = (-Yk .* A) - ((1 - Yk) .* B);
	J = J + sum(temp);
	
	
	% --------------   Part 2   -------------- %
	
	%Trim the bias unit off of everything
	Theta2x = Theta2(:, 2:end);
	a2 = a2(2:end);
	
	Theta1x = Theta1(:, 2:end);
	ex = ex(:, 2:end);
	
	
	

	%Calculate error in output layer
	d3 = a3 .- Yk;
	
	
	%Calculate error in layer #2
	d2 = (Theta2x' * d3) .* (a2 .* (1 - a2));
	
	
	%Take dot product of layer2 error and layer2 output, but remove bias neuron
	% D2 = D2 + (d2(2:end) .* a2(2:end));
	D2 = D2 + (d2 * (a2)');
	printf("D2\n")
	size(d2)
	size(D2)
	
	printf("D1\n")
	size(ex)
	size(Theta1x')
	size(d2)
	
	size((Theta1x' * d2)')
	size((ex .* (1 - ex))')
	
	%Calculate error in layer #1
	 d1 = (Theta1x' * d2) * (ex .* (1 - ex));
	%d1 = (ex .* (1 - ex)) * (Theta1x' * d2);
	
	size(d1)
	f
	
	
	%Take dot product of layer1 error and layer1 output, but remove bias neuron
	D1 = D1 + (d1 .* ex);
	
	
	% ---------------------------------------- %
	
	
	
	
endfor

size(D2)
size(D1)



J = J / m;
Theta2_grad = D2/m;
Theta1_grad = D1/m;





% --- Add Regularization --- %

%nn_params contains all theta
reg = sum(nn_params .^ 2);
reg = (lambda / (2 * m)) * reg ;

J = J + reg;




% ---------------------------------------- %


% --------------   Part 3   -------------- %
% ---------------------------------------- %












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
