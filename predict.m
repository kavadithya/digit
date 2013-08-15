function p = predict(nn_params,layers, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = layers(end);%size(Theta3, 1);
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h = X;
for u=1:size(layers,2) - 1
	h = [ones(m,1) h];
	Temp_theta = nnGenTheta(nn_params,u+1,layers);
	h = sigmoid(h*(Temp_theta'));
end	


% h1 = sigmoid([ones(m, 1) X] * Theta1');
% h2 = sigmoid([ones(m, 1) h1] * Theta2');
% h3 = sigmoid([ones(m,1) h2]*Theta3');
[dummy, p] = max(h, [], 2);
for i=1:size(p,1)
	if p(i) == 10
		p(i) = 0;
	endif
end
% =========================================================================


end
