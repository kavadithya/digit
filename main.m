%clear ; 
close all; clc
input_layer_size  = 28*28; 
num_labels = 10;

%data = load('train3.csv');
y = data(1:12000,1);
X = data(1:12000,2:end);
Xval = data(12001:14000,2:end);
yval = data(12001:14000,1);
Xtest = data(14001:16000,2:end);
ytest = data(14001:16000,1);
fprintf('\nDone getting data and loading X,y ...\n')
m = size(X, 1);
m_test = size(Xval,1);
layers = [784,200,10];

lambda = [3 3.5 4 4.5 5];
layer_size = [250];
prediction = zeros(size(lambda,2),size(layers,2));
costmat = zeros(size(lambda,2),size(layers,2));
for j=1:size(lambda,2)
	for k=1:size(layer_size,2)
		nn_params = debugInitializeWeights(layers);
		fprintf('\nTraining Neural Network... \n')
		options = optimset('MaxIter', 200);
		costFunction  = @(p)nnCostFunction_later(p,layers,X,y,lambda(j));
		[nn_params, cost] = fmincg(costFunction,nn_params, options);
		pred = predict(nn_params,layers,Xval);
		fprintf('\nFor lambda = %f, hidden_layers = %f and layer_size = %f, training Set Accuracy: %f\n',lambda(j),size(layers,2)-2,layer_size(k),mean(double(pred == yval)) * 100);
		prediction(j,k) = mean(double(pred == yval)) * 100;
		costmat(j,k) = cost(end);
	end
end

%save nn_params_200_1point9.mat nn_params
vec = [1:m_test];
Jtrain = zeros(size(vec,2),1);
Jval = zeros(size(vec,2),1);
for i=1:size(vec,2)
	Jtrain(i) = nnCostAlone(nn_params,layers,X(1:i,:),y(1:i,:),0);
	Jval(i) = nnCostAlone(nn_params,layers,Xval(1:i,:),yval(1:i,:),0);
end
plot(vec,Jtrain,vec,Jval);

% subplot(5,2,j)




%mesh(lambda,layers,costmat);
				 
				 
				 
% save Theta1.mat Theta1;
% save Theta2.mat Theta2;
% save Theta3.mat Theta3;
% % load('Theta1.mat');
% % load('Theta2.mat');
% % load('Theta3.mat');
% % nn_params = [Theta1(:);Theta2(:)];






