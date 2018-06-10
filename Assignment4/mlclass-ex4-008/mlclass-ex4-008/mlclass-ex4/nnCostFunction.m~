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
w = size(X, 2);
         
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
repb = zeros(1,num_labels);
yvec = zeros(m,num_labels);
repb=1:num_labels;
yvec=repmat(repb,m,1);
yvec=(yvec==y);
a1 = zeros(size(X,1),size(X,2)+1);
a1 = zeros(size(X,1),size(X,2)+1);
k2 = zeros(size(X,1),size(Theta1,1));
a2 = zeros(size(X,1),size(Theta2,2));
a3 = zeros(size(X,1),size(Theta2,1));
k3 = zeros(size(Theta2,1),size(X,1));


sums = 0;
	a1 = [ones(m,1),X];
	for i=1:m,
	k2(i,:) = (Theta1*a1(i,:)');
	end;
	a2 = [ones(m,1),sigmoid(k2)];
	for i=1:m,
	a3(i,:) = sigmoid(Theta2*a2(i,:)');
	end;
	
	for i=1:m,
	sums = sums - (yvec(i,:)*log(a3(i,:))') - ((1-yvec(i,:))*log(1-a3(i,:))');
	end;

	
regular = 0;
for i=1:size(Theta1,1),
	for j=2:size(Theta1,2),
		regular = regular + Theta1(i,j)*Theta1(i,j);
	end;
end;

for i=1:size(Theta2,1),
	for j=2:size(Theta2,2),
		regular = regular + Theta2(i,j)*Theta2(i,j);
	end;
end;
J = sums/m + regular*lambda/(2*m);
	
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

d3 = zeros(size(Theta2,1),1);
d2 = zeros(size(Theta1,1),1);
d2temp = zeros(size(Theta2,2),1);
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
for i=1:m,
	
	d3 = (a3(i,:) - yvec(i,:))';
	d2temp = ((Theta2)' * d3);
	d2 = d2temp(2:end);
	d2 = d2.*sigmoidGradient(k2(i,:)');
	D1 = D1 + (d2*a1(i,:));
	D2 = D2 + (d3*a2(i,:)); 		
end;

	%Theta1_grad(:,1) = D1(:,1)/m;
	%Theta2_grad(:,1) = D2(:,1)/m;
%for i=2:(size(D1,2)),
%	Theta1_grad(:,i) = D1(:,i)/m + (lambda/m)*Theta1(:,i);
%end;
%for i=2:(size(D2,2)),
%	Theta2_grad(:,i) = D2(:,i)/m + (lambda/m)*Theta2(:,i);
%end;
mask=ones(size(Theta1));
mask(:,1)=0;
%%grad=1./m*((sigmoid(X*theta)-y)'*X)+lambda*(theta .* mask)/m;
Theta1_grad=D1/m+lambda*(Theta1 .*mask)/m;
mask=ones(size(Theta2));
mask(:,1)=0;
Theta2_grad=D2/m+lambda*(Theta2 .*mask)/m;

	
	


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
