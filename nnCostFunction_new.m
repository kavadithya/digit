function [J grad] = nnCostFunction_new(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size1, hidden_layer_size2, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), ...
                 hidden_layer_size1, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size1 * (input_layer_size + 1))): hidden_layer_size1*(input_layer_size + 1) + hidden_layer_size2*(hidden_layer_size1 + 1)), ...
                 hidden_layer_size2, (hidden_layer_size1 + 1));
Theta3 = reshape(nn_params((1 + hidden_layer_size1*(input_layer_size + 1) + (hidden_layer_size2 * (hidden_layer_size1 + 1))):end), ...
                 num_labels, (hidden_layer_size2 + 1));


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

X = [ones(m,1) X];
Z2 = Theta1*X';
A2 = sigmoid(Z2);
A2 = [ones(1,size(A2,2)); A2];
Z3 = Theta2*A2;
A3 = sigmoid(Z3);
A3 = [ones(1,size(A3,2));A3];
Z4 = Theta3*A3;
hTheta = sigmoid(Z4);
lh = log(hTheta);
lh_1 = log(1-hTheta);
temp = zeros(num_labels,1);

for i=1:m
    temp_x1 = lh(:,i);
	temp_x2 = lh_1(:,i);
	temp_y = zeros(num_labels,1);
	if y(i) == 0
		y(i) = 10;
	endif
	temp_y(y(i)) = 1;
	J += temp_x1'*temp_y + temp_x2'*(1-temp_y);
end
J = (-1/m)*J;

nn_new = [Theta1(:,2:end)(:);Theta2(:,2:end)(:); Theta3(:,2:end)(:)];

J += (lambda/(2*m))*sum(nn_new.^2);

Delta1 = zeros(hidden_layer_size1, input_layer_size + 1);
Delta2 = zeros(hidden_layer_size2, hidden_layer_size1 + 1);
Delta3 = zeros(num_labels, hidden_layer_size2 + 1);

for t=1:m
    a1 = X(t,:);
	z2 = Z2(:,t);%
	a2 = A2(:,t);%;%
	z3 = Z3(:,t);%;%
	a3 = A3(:,t); %;%
	z4 = Z4(:,t);%;%
	h = hTheta(:,t);%;%
	temp = zeros(num_labels,1);
	if y(t) == 0
		y(t) = 10;
	endif
	temp(y(t)) = 1;
	delta4 = h - temp;
	delta3 = ((Theta3')*delta4)(2:end).*sigmoidGradient(z3);
	delta2 = ((Theta2')*delta3)(2:end).*sigmoidGradient(z2);
	Delta1 += delta2*(a1);
	Delta2 += delta3*(a2');
	Delta3 += delta4*(a3');
end

Theta1_grad(:,1) = (1/m)*Delta1(:,1);
Theta1_grad(:,2:end) = (1/m)*Delta1(:,2:end) + (lambda/m)*Theta1(:,2:end); %(lambda/m)*sum([Theta1(:,2:end)(:)]);
Theta2_grad(:,1) = (1/m)*Delta2(:,1);
Theta2_grad(:,2:end) = (1/m)*Delta2(:,2:end) + (lambda/m)*Theta2(:,2:end); %(lambda/m)*sum([Theta2(:,2:end)(:)]);
Theta3_grad(:,1) = (1/m)*Delta3(:,1);
Theta3_grad(:,2:end) = (1/m)*Delta3(:,2:end) + (lambda/m)*Theta3(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:)];


end
