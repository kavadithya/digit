function [nn] = debugInitializeWeights(layers)
%DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
%incoming connections and fan_out outgoing connections using a fixed
%strategy, this will help you later in debugging
%   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
%   of a layer with fan_in incoming connections and fan_out outgoing 
%   connections using a fix set of values
%
%   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
%   the first row of W handles the "bias" terms
%

nn = [];

for i=1:size(layers,2) -1 
	switch i
		case 8
			W = zeros(layers(i+1),1+layers(i));
			W = reshape(sin(1:numel(W)), size(W)) / 10;
			nn = [nn(:);W(:)];
		case 7
			W = zeros(layers(i+1),1+layers(i));
			W = reshape(sin(1:numel(W)), size(W)) / 10;
			nn = [nn(:);W(:)];
		case 6
			W = zeros(layers(i+1),1+layers(i));
			W = reshape(sin(1:numel(W)), size(W)) / 10;
			nn = [nn(:);W(:)];
		case 5
			W = zeros(layers(i+1),1+layers(i));
			W = reshape(sin(1:numel(W)), size(W)) / 10;
			nn = [nn(:);W(:)];
		case 4
			W = zeros(layers(i+1),1+layers(i));
			W = reshape(sin(1:numel(W)), size(W)) / 10;
			nn = [nn(:);W(:)];
		case 3 
			W = zeros(layers(i+1),1+layers(i));
			W = reshape(sin(1:numel(W)), size(W)) / 10;
			nn = [nn(:);W(:)];
		case 2
			W = zeros(layers(i+1),1+layers(i));
			W = reshape(sin(1:numel(W)), size(W)) / 10;
			nn = [nn(:);W(:)];
		case 1
			W = zeros(layers(i+1),1+layers(i));
			W = reshape(sin(1:numel(W)), size(W)) / 10;
			nn = [nn(:);W(:)];
	endswitch
end









% Set W to zeros


% Initialize W using "sin", this ensures that W is always of the same
% values and will be useful for debugging


% =========================================================================

end
