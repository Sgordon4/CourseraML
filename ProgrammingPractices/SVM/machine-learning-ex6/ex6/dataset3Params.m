function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


%Options to choose C and sigma from
options = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
rows = rows(options);

%Create a matrix to hold the error for each model
errors = zeros(rows, rows);

%For every option
for c = 1:rows 
	for sig = 1: rows	
		
		%Compute and print combination number
		combo = (c - 1) * rows + sig;
		fprintf(['Training combination number %d:'], combo);
		
		%Assign the values of C and sigma to test
		C = options(c);
		sigma = options(sig);
		
		
		%Train the model with the respective C and sigma
		model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
		
		%Grab the predictions from the trained model
		predictions = svmPredict(model, Xval);
		
		%Find the cost
		error = mean(double(predictions ~= yval));
		
		%Store the error in its respective position
		errors(c, sig) = error;
		
	end
end


%Find the position of the lowest error
minVal = min(min(errors));
[i, j] = find(errors == minVal);

%Assign the respective values to C and sigma
C = options(i);
sigma = options(j);


% =========================================================================

end
