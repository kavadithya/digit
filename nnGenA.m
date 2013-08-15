function [a nn_new] = nnGenA(nn_params,layers,subscript,X)

nn_new = [];
z = [];
if subscript==1
	a = X';
	a  = [ones(1,size(a,2)); a];
	
endif
	
for i=2:subscript
	if i==2
		a = (X)';
		a  = [ones(1,size(a,2)); a];
	endif
	Theta_temp =  nnGenTheta(nn_params,i,layers);%reshape(nn_params(start_param:end_param),layers(i),layers(i-1) + 1);
	%a  = [ones(1,size(a,2)); a];
	z = Theta_temp*a;
	a = sigmoid(z);
	if i!= size(layers,2)
		a  = [ones(1,size(a,2)); a];
	end
	nn_new = [nn_new; Theta_temp(:,2:end)(:)];
end

end