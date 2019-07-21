function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%


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

% Compute final cost
J = center / m;

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

%------------------------------------------%


% =============================================================

end
