from __future__ import division
from numpy import *

import numpy
input_layer_size  = 28*28
num_labels = 10
# data = genfromtxt('train3.csv',delimiter=',')
# y = data[0:100,0]
# X = data[0:100,1:]
# Xval = data[12001:14000,2:]
# yval = data[12001:14000,1]
# Xtest = data[14001:16000,2:]
# ytest = data[14001:16000,1]
# print '\nDone getting data and loading X,y ...\n'
# m = X.shape[0]
# m_test = Xval.shape[0]
# layers = array([784,200,10])
# Lambda = array([3,3.5,4,4.5,5])
# layer_size = array([250])

def arraysize(ar,dim):
	try:
		l = ar.shape[dim]
	except:
		l = ar.shape[dim-1]
	return l

def debugInitializeWeights(layers):
	Theta = []
	for i in range(0,arraysize(layers,1) -1):
		theta_temp = random.random((layers[i+1],layers[i] + 1))
		Theta.append(theta_temp)
	return Theta

def sigmoid(arr):
	arr = 1.0 / (1.0 + exp(-arr));
	return arr
	
def sigmoidGradient(z):
	g = sigmoid(z)*(1 - sigmoid(z));
	return g
	
def computeNumericalGradient(theta,layers,X,y,lambda_temp):
	numgrad = []
	perturb = []
	for i in theta:
		numgrad.append(i*0)
		perturb.append(i*0)
	e = 1e-4
#	print theta
	for p  in range(len(theta)):
		# Set perturbation vector
		for q in range(len(theta[p])):
			for k in range(len(perturb[p][q])):
				perturb[p][q][k] = e;
				temp1 = theta
				temp1[p][q][k] -= 1e-4

				# for i in range(len(theta)):
					# print temp[i]
				(loss1,junk) = nnCostFunction_later(temp1,layers,X,y,lambda_temp)
	#			for i in range(len(theta)):
	#				print temp[i]
				temp2 = theta
				temp2[p][q][k] += 1e-4	
				(loss2,junk) = nnCostFunction_later(temp2,layers,X,y,lambda_temp)
				# Compute Numerical Gradient
				numgrad[p][q][k]  = (loss2 - loss1) / (2*e)
#				print loss1, loss2, type(numgrad[p][q][k]), numgrad[p][q][k]
				perturb[p][q][k] = 0
#	print numgrad
	return numgrad
		

	
def checkNNGradients(lambda_temp):
	input_layer_size = 3
	hidden_layer_size = 5
	num_labels = 3
	m = 5
	layers = array([input_layer_size,hidden_layer_size,num_labels])
	Theta = debugInitializeWeights(layers)
	X  = random.random((m, input_layer_size))
	y  = numpy.random.randint(1,num_labels,m)
	(cost1,grad1) = nnCostFunction_later(Theta,layers,X,y,0)
	numgrad = computeNumericalGradient(Theta,layers,X,y,0);
	# Visually examine the two gradient computations.  The two columns
	# you get should be very similar. 
	for i in range(len(numgrad)):
		print numgrad[i] - grad1[i]
	# print numgrad
	# print grad1


	
	
	
def nnCostFunction_later(Theta,layers,X,y,lambda_temp):
	m = X.shape[0]
	num_labels = layers[-1]
	J = 0
#	nn_new = array([])
	A = [X.transpose()]
	Z = [0]
	Delta = []
	Grad = []
	for i in range(1,arraysize(layers,1)):
		A[i-1] = vstack(( ones((1,arraysize(A[i-1],1))), A[i-1]))
		# print i, Theta[i-1].shape, A[i-1].shape
		Z.append(dot(Theta[i-1],A[i-1]))
		A.append(sigmoid(Z[i]))
	H = A[-1]
	lh = log(H)
	lh_1 = log(1-H)
	temp = zeros((num_labels,1))
	for i in range(1,m):
		temp_x1 = lh[:,i]
		temp_x2 = lh_1[:,i]
		temp_y = zeros((num_labels,1))
		# if y[i] == 0:
			# y[i] = num_labels
		temp_y[y[i]] = 1
		J += dot(temp_x1.transpose(),temp_y) + dot(temp_x2.transpose(),(1-temp_y))
	J = (-1/m)*J;
	for i in range(len(Theta)):
		J += dot((lambda_temp/(2*m)),sum(Theta[i]*Theta[i]))
	for i in range(0,arraysize(layers,1) - 1):
		Delta.append(zeros((layers[i+1],layers[i] + 1)))
	for t in range(0,m-1):
		a1 = A[0][:,t]
		h = H[:,t]
		temp = zeros(h.shape)
		temp[y[t]] = 1
		delta = [0]
		for i in range(0,arraysize(layers,1) - 1): ## size(layers,2) == 9, so i runs from 0 to 7 => d runs from 9 to 2
			d = arraysize(layers,1) - i
			if d == arraysize(layers,1):
				delta.insert(1,h - temp)
			else:
				z = Z[d - 1][:,t]
				a = A[d-1][:,t]
				delta.insert(1,dot(Theta[d-1].transpose(),delta[-1])[1:]*sigmoidGradient(z))
		for c in range(0,arraysize(layers,1)-1):
			Delta[c] += dot(array([delta[c+1]]).T,A[c][:,t][newaxis])
	for i in range(0,arraysize(layers,1) - 1):
		Grad.append(zeros(shape(Theta[i])))
		Grad[i][:,0] = (1/m)*Delta[i][:,0]
		Grad[i][:,1:] = (1/m)*Delta[i][:,1:] + (lambda_temp/m)*Theta[i][:,1:]
#		print Grad[i]
	# for i in Delta:
		# print i
	# for i in Grad:
		# print i
	return (J,Grad)
	
	

checkNNGradients(1.1)	

# for j in range(1,arraysize(Lambda,1)+1):
	# for k in range(1,arraysize(layer_size,1)+1):
		# Theta = debugInitializeWeights(layers)
		# print '\nTraining Neural Network... '
		# options = optimset('MaxIter', 200);
		# costFunction  = @(p)nnCostFunction_later(p,layers,X,y,lambda(j));
		#(n_params, cost) = fmincg(costFunction,nn_params, options);
		# pred = predict(nn_params,layers,Xval);
		# fprintf('\nFor lambda = %f, hidden_layers = %f and layer_size = %f, training Set Accuracy: %f\n',lambda(j),size(layers,2)-2,layer_size(k),mean(double(pred == yval)) * 100);
		# prediction(j,k) = mean(double(pred == yval)) * 100;
		# costmat(j,k) = cost(end);

