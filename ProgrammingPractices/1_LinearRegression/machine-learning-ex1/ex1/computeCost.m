function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


C = 1/ ( 2 * size(X,1));			%1/2m

h = sum(theta' .* X, 2);			%Create the hypothesis, make it an m dimensional column vector

S = (h - y) .^2;					%(h(i) - y(i))^2


J = C * sum(S);						%Sum the m dimensional column vector, and multiply it by 1/2m
%J = (1/2m) * sum[1 -> m](h(i) - y(i))^2

% =========================================================================

end
