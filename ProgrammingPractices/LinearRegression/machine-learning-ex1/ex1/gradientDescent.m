function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


	% disp('-----------------------------------')

	C = 1 / m;							%1/m	--	1/2m gives incorrect results, too low
	C = alpha * C;						%a(1/m)
	
	%---------------- Creating (d/dx)J ----------------%
	hypothesis = sum(theta' .* X, 2);	%Create the hypothesis, compress it into an m dimensional column vector
	
	cost = hypothesis - y;				%Find the difference between our prediction and the real value
	
	costScalars = cost .* X;			%Scale by each value of X
	
	scalarsSum = sum(costScalars, 1);	%Flatten the scalars into an n dimensional row vector
	%--------------------------------------------------%
	
	change = scalarsSum .* C;			%a(1/m)*(d/dx)J
	
	theta = theta - change';				%Finally, modify theta


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
