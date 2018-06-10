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
hypothesis = sigmoid(X*theta);
sqrerrors = -y'*log(hypothesis) -  (1-y)'*log(1-hypothesis);
k = (1/m) * sum(sqrerrors);
p = size(theta,1);

sum = 0;
for i=2:p,
	sum = sum + theta(i)^2;
end;
sum = 	(lambda/(2*m))*sum;
J = sum + k;

v = zeros(size(theta));
predictions = hypothesis - y;
v = (1/m)*X'*predictions;
theta2 = zeros(size(theta));
for i=2:p,
	theta2(i) = theta(i);
end;
grad = v + (lambda/m)*theta2;







% =============================================================

end
