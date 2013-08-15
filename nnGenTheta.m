function [Theta] = nnGenTheta(nn_params,subscript, layers )

for i=2:subscript
	if i==2
		start_param = 1;
		end_param = layers(2)*(layers(1) + 1) ;
	else
		start_param = 1 + end_param;
		end_param = start_param - 1 + (layers(i)*(layers(i-1) + 1));
	endif
end

Theta = reshape(nn_params(start_param:end_param),layers(i),layers(i-1) + 1);

end
                                  