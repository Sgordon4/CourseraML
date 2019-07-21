function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



%---------- Computing Hypothesis ----------%

% Prepare sigmoid's food
z = theta' .* X;
z = sum(z, 2);

% Feed sigmoid and grab the resulting hypothesis before it runs away with it
h = sigmoid(z);

%------------------------------------------%



%------------- Computing Cost -------------%

% Build the two halves of the cost function
A = log(h);
B = log(1 - h);

% Slap them together
center = (-y .* A) - ((1.-y) .* B);
% And squish them flat
center = sum(center, 1);

% Compute cost without regularization
J = center / (m);



%      -- Computing Regularization --

% Sum the squares of theta (except theta(1))
sums = sum(theta(2:end, :) .^ 2, 1);

reg = lambda * sums / (m);

% Complete J
J = J + reg;

%------------------------------------------%



%----------- Computing Gradient -----------%

% Find difference between our guess and reality (often dissapointing)
diff = h - y;

% Scale by each value of X
scaled = diff .* X;

% Squish the scalars into an n dimensional row vector
vector = sum(scaled, 1);

% And divide it all by m
vector = vector ./ m;

% Now flip it to match theta
grad = vector';



%      -- Computing Regularization --

% Replace theta(1) with 0 so it is not penalized
theta_reg = [0; theta(2:end, :)];

% Compute regression matrix
reg = (lambda / m) .* theta_reg;

% Add regularization onto our cost gradient
grad = grad .+ reg;



%------------------------------------------%



% =============================================================

end
