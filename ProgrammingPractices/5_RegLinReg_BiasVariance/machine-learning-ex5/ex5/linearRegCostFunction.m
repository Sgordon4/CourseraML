function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% size(X)
% X
% size(y)
% y
% size(theta)
% theta



%-- Compute Cost --%

%Calculate Hypothesis
h = X * theta;

%Calculate Differences
diff = (h .- y) .^ 2;

%Find average of differences
J = sum(diff, 1) / (2*m);


%-- Now Regularize --%

%Replace theta_0 with 0 so it is not regularized
thetax = [0;theta(2:end)];

reg = sum(thetax .^ 2, 1) * lambda / (2 * m);

J = J + reg;


%-- Compute Gradient --%

%Find the difference between predicted and real
diff = (h - y);

%Multiply by each data point
grad = diff .* X;

%Sum vertically and average
grad = sum(grad, 1) / m;


%-- Now Regularize --%

%Only regularize theta1 (thetax has theta0 set to 0)
reg = (lambda * thetax) / m;

grad = grad + reg';


% =========================================================================

grad = grad(:);

end
