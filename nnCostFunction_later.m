function [J grad] = nnCostFunction_later(nn_params, ...
                                   layers, ...
                                   X, y, lambda)

%%% Supports till size(layers) == 9
								   
m = size(X, 1);
num_labels = layers(end);         
J = 0;
nn_new = [];

for i=2:size(layers,2)
	switch i
		case 2
			A1 = X';
			start_param = 1;
			end_param = layers(2)*(layers(1) + 1) ;
			A1  = [ones(1,size(A1,2)); A1];
			Theta_temp = Theta1 = reshape(nn_params(start_param:end_param),layers(i),layers(i-1) + 1);
			Z2 = Theta_temp*A1;
			A2 = sigmoid(Z2);
			A2 = [ones(1,size(A2,2)); A2];
			nn_new = [nn_new; Theta_temp(:,2:end)(:)];
		case 3
			start_param = 1 + end_param;
			end_param = start_param - 1 + (layers(i)*(layers(i-1) + 1));
			Theta_temp = Theta2 = reshape(nn_params(start_param:end_param),layers(i),layers(i-1) + 1);
			Z3 = Theta_temp*A2;
			A3 = sigmoid(Z3);
			if i==size(layers,2)
				H = A3;
			endif
			A3 = [ones(1,size(A3,2)); A3];
			nn_new = [nn_new; Theta_temp(:,2:end)(:)];
		case 4
			start_param = 1 + end_param;
			end_param = start_param - 1 + (layers(i)*(layers(i-1) + 1));
			Theta_temp = Theta3 = reshape(nn_params(start_param:end_param),layers(i),layers(i-1) + 1);
			Z4 = Theta_temp*A3;
			A4 = sigmoid(Z4);
			if i==size(layers,2)
				H = A4;
			endif
			A4 = [ones(1,size(A4,2)); A4];
			nn_new = [nn_new; Theta_temp(:,2:end)(:)];
		case 5
			start_param = 1 + end_param;
			end_param = start_param - 1 + (layers(i)*(layers(i-1) + 1));
			Theta_temp = Theta4 = reshape(nn_params(start_param:end_param),layers(i),layers(i-1) + 1);
			Z5 = Theta_temp*A4;
			A5 = sigmoid(Z5);
			if i==size(layers,2)
				H = A5;
			endif
			A5 = [ones(1,size(A5,2)); A5];
			nn_new = [nn_new; Theta_temp(:,2:end)(:)];
		case 6
			start_param = 1 + end_param;
			end_param = start_param - 1 + (layers(i)*(layers(i-1) + 1));
			Theta_temp = Theta5 = reshape(nn_params(start_param:end_param),layers(i),layers(i-1) + 1);
			Z6 = Theta_temp*A5;
			A6 = sigmoid(Z6);
			if i==size(layers,2)
				H = A6;
			endif
			A6 = [ones(1,size(A6,2)); A6];
			nn_new = [nn_new; Theta_temp(:,2:end)(:)];
		case 7
			start_param = 1 + end_param;
			end_param = start_param - 1 + (layers(i)*(layers(i-1) + 1));
			Theta_temp = Theta6 = reshape(nn_params(start_param:end_param),layers(i),layers(i-1) + 1);
			Z7 = Theta_temp*A6;
			A7 = sigmoid(Z7);
			if i==size(layers,2)
				H = A7;
			endif
			A7 = [ones(1,size(A7,2)); A7];
			nn_new = [nn_new; Theta_temp(:,2:end)(:)];
		case 8
			start_param = 1 + end_param;
			end_param = start_param - 1 + (layers(i)*(layers(i-1) + 1));
			Theta_temp = Theta7 = reshape(nn_params(start_param:end_param),layers(i),layers(i-1) + 1);
			Z8 = Theta_temp*A7;
			A8 = sigmoid(Z8);
			if i==size(layers,2)
				H = A8;
			endif
			A8 = [ones(1,size(A8,2)); A8];
			nn_new = [nn_new; Theta_temp(:,2:end)(:)];
		case 9 
			start_param = 1 + end_param;
			end_param = start_param - 1 + (layers(i)*(layers(i-1) + 1));
			Theta_temp = Theta8 = reshape(nn_params(start_param:end_param),layers(i),layers(i-1) + 1);
			Z9 = Theta_temp*A8;
			A9 = sigmoid(Z9);
			if i==size(layers,2)
				H = A9;
			endif
			A9 = [ones(1,size(A9,2)); A9];
			nn_new = [nn_new; Theta_temp(:,2:end)(:)];

	endswitch
%	Theta_temp =  nnGenTheta(nn_params,i,layers);%reshape(nn_params(start_param:end_param),layers(i),layers(i-1) + 1);
	% a  = [ones(1,size(a,2)); a];
	% z = Theta_temp*a;
	% a = sigmoid(z);
end
lh = log(H);
lh_1 = log(1-H);
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
J += (lambda/(2*m))*sum(nn_new.^2);

for u=1:size(layers,2) -1 
	switch u
		case 8
			Delta8 = zeros(layers(u+1),layers(u) + 1);
		case 7
			Delta7 = zeros(layers(u+1),layers(u) + 1);
		case 6
			Delta6 = zeros(layers(u+1),layers(u) + 1);
		case 5
			Delta5 = zeros(layers(u+1),layers(u) + 1);
		case 4
			Delta4 = zeros(layers(u+1),layers(u) + 1);
		case 3 
			Delta3 = zeros(layers(u+1),layers(u) + 1);
		case 2
			Delta2 = zeros(layers(u+1),layers(u) + 1);
		case 1
			Delta1 = zeros(layers(u+1),layers(u) + 1);
	endswitch
end


for t=1:m
	temp = zeros(num_labels,1);
	if y(t) == 0
		y(t) = 10;
	endif
	temp(y(t)) = 1;
    a1 = A1(:,t);
	h = H(:,t);
	for i=0:size(layers,2) - 2 % size(layers,2) == 9, so i runs from 0 to 7 => d runs from 9 to 2
		d = size(layers,2) - i;
		
		switch d
			case 9
				delta9 = h - temp;
			case 8
				if d == size(layers,2)
					delta8 = h - temp;
				else
					z8 = Z8(:,t);
					a8 = A8(:,t);
					delta8 = ((Theta8')*delta9)(2:end).*sigmoidGradient(z8);
				endif
			case 7
				if d == size(layers,2)
					delta7 = h - temp;
				else
					z7 = Z7(:,t);
					a7 = A7(:,t);
					delta7 = ((Theta7')*delta8)(2:end).*sigmoidGradient(z7);
				endif
			case 6
				if d == size(layers,2)
					delta6 = h - temp;
				else
					z6 = Z6(:,t);
					a6 = A6(:,t);
					delta6 = ((Theta6')*delta7)(2:end).*sigmoidGradient(z6);
				endif
			case 5
				if d == size(layers,2)
					delta5 = h - temp;
				else
					z5 = Z5(:,t);
					a5 = A5(:,t);
					delta5 = ((Theta5')*delta6)(2:end).*sigmoidGradient(z5);
				endif
			case 4
				if d == size(layers,2)
					delta4 = h - temp;
				else
					z4 = Z4(:,t);
					a4 = A4(:,t);
					delta4 = ((Theta4')*delta5)(2:end).*sigmoidGradient(z4);
				endif
			case 3
				if d == size(layers,2)
					delta3 = h - temp;
				else
					z3 = Z3(:,t);
					a3 = A3(:,t);
					delta3 = ((Theta3')*delta4)(2:end).*sigmoidGradient(z3);
				endif
			case 2
				if d == size(layers,2)
					delta2 = h - temp;
				else
					z2 = Z2(:,t);
					a2 = A2(:,t);
					delta2 = ((Theta2')*delta3)(2:end).*sigmoidGradient(z2);
				endif
		endswitch
	end
	for c=1:(size(layers,2) - 1)
		switch c
			case 1
				Delta1 += delta2*(a1');
			case 2
				Delta2 += delta3*(a2');
			case 3
				Delta3 += delta4*(a3');
			case 4
				Delta4 += delta5*(a4');
			case 5
				Delta5 += delta6*(a5');
			case 6
				Delta6 += delta7*(a6');
			case 7
				Delta7 += delta8*(a7');
			case 8
				Delta8 += delta9*(a8');
		endswitch
	end
end
grad = [];

for i=1:(size(layers,2)-1)
	switch i
		case 8
			Theta8_grad = zeros(size(Theta8));
			Theta8_grad(:,1) = (1/m)*Delta8(:,1);
			Theta8_grad(:,2:end) = (1/m)*Delta8(:,2:end) + (lambda/m)*Theta8(:,2:end); %(lambda/m)*sum([Theta1(:,2:end)(:)]);
			grad = [grad(:); Theta8_grad(:)];

		case 7
			Theta7_grad = zeros(size(Theta7));
			Theta7_grad(:,1) = (1/m)*Delta7(:,1);
			Theta7_grad(:,2:end) = (1/m)*Delta7(:,2:end) + (lambda/m)*Theta7(:,2:end); %(lambda/m)*sum([Theta1(:,2:end)(:)]);
			grad = [grad(:); Theta7_grad(:)];

		case 6
			Theta6_grad = zeros(size(Theta6));
			Theta6_grad(:,1) = (1/m)*Delta6(:,1);
			Theta6_grad(:,2:end) = (1/m)*Delta6(:,2:end) + (lambda/m)*Theta6(:,2:end); %(lambda/m)*sum([Theta1(:,2:end)(:)]);
			grad = [grad(:); Theta6_grad(:)];

		case 5
			Theta5_grad = zeros(size(Theta5));
			Theta5_grad(:,1) = (1/m)*Delta5(:,1);
			Theta5_grad(:,2:end) = (1/m)*Delta5(:,2:end) + (lambda/m)*Theta5(:,2:end); %(lambda/m)*sum([Theta1(:,2:end)(:)]);
			grad = [grad(:); Theta5_grad(:)];

		case 4
			Theta4_grad = zeros(size(Theta4));
			Theta4_grad(:,1) = (1/m)*Delta4(:,1);
			Theta4_grad(:,2:end) = (1/m)*Delta4(:,2:end) + (lambda/m)*Theta4(:,2:end); %(lambda/m)*sum([Theta1(:,2:end)(:)]);
			grad = [grad(:); Theta4_grad(:)];

		case 3
			Theta3_grad = zeros(size(Theta3));
			Theta3_grad(:,1) = (1/m)*Delta3(:,1);
			Theta3_grad(:,2:end) = (1/m)*Delta3(:,2:end) + (lambda/m)*Theta3(:,2:end); %(lambda/m)*sum([Theta1(:,2:end)(:)]);
			grad = [grad(:); Theta3_grad(:)];

		case 2
			Theta2_grad = zeros(size(Theta2));
			Theta2_grad(:,1) = (1/m)*Delta2(:,1);
			Theta2_grad(:,2:end) = (1/m)*Delta2(:,2:end) + (lambda/m)*Theta2(:,2:end); %(lambda/m)*sum([Theta1(:,2:end)(:)]);
			grad = [grad(:); Theta2_grad(:)];
		case 1
			Theta1_grad = zeros(size(Theta1));
			Theta1_grad(:,1) = (1/m)*Delta1(:,1);
			Theta1_grad(:,2:end) = (1/m)*Delta1(:,2:end) + (lambda/m)*Theta1(:,2:end); %(lambda/m)*sum([Theta1(:,2:end)(:)]);
			grad = [grad(:); Theta1_grad(:)];
	endswitch
end

% % -------------------------------------------------------------

% % =========================================================================

% % Unroll gradients
% grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:)];
%size(grad)

end
