### Digits recognition example for ffnet ###

# Training file (data/ocr.dat) contains 68 patterns - first 58
# are used for training and last 10 are used for testing.
# Each pattern contains 64 inputs which define 8x8 bitmap of
# the digit and last 10 numbers are the targets (10 targets for 10 digits).
# Layered network architecture is used here: (64, 10, 10, 10).
from __future__ import division

import numpy
from ffnet import ffnet, mlgraph, readdata

def predict(pred,y):
	c = 0
	for i in range(len(pred)):
		if numpy.argmax(pred[i]) == y[i]:
			c += 1
#			print 'good ', pred[i], y[i]
		else: 
#			print 'bad ', pred[i], y[i]
			pass
	print 'c = ', c,' len = ', len(pred)
	return (c/len(pred))*100

# Generate standard layered network architecture and create network
conec = mlgraph((400,200,10))
net = ffnet(conec)

# Read data file
print "READING DATA..."
data = readdata( 'data.csv')
numpy.random.shuffle(data)
X = data[:,1:]
y =  data[:,0]#first 64 columns - bitmap definition
input = X
target = numpy.ndarray((input.shape[0],10))
for i in range(len(y)):
	target[i] = numpy.zeros((1,10))
	if y[i] == 10:
		y[i] = 0
	target[i][y[i]] = 1

# Train network with scipy tnc optimizer - 58 lines used for training
print "TRAINING NETWORK..."
net.train_tnc(input[:4500], target[:4500], maxfun = 3000, messages=1)
#net.train_cg(input[:100],target[:100],maxiter=2000)
pred = net.call(input[:4500])

print 'PREDICTION'
print 'Accuracy = ',predict(pred,y[:4500])
print 'TESTING'
y_test = net.call(input[4501:4900])
print 'Accuracy = ', predict(y_test,y[4501:4900])
# for i in range(len(y_test)):
	# print numpy.argmax(y_test[i]), y[101+i]


# # Test network - remaining 10 lines used for testing
# # print
# # print "TESTING NETWORK..."
# # output, regression = net.test(input[100:200], target[100:200], iprint = 2)
# #print target[100:200] - output
# #print regression
# # ############################################################
# # # Make a plot of a chosen digit along with the network guess
# # try:
    # # from pylab import *
    # # from random import randint

    # # digitpat = randint(58, 67) #Choose testing pattern to plot

    # # subplot(211)
    # # imshow(input[digitpat].reshape(8,8), interpolation = 'nearest')

    # # subplot(212)
    # # N = 10  # number of digits / network outputs
    # # ind = arange(N)   # the x locations for the groups
    # # width = 0.35       # the width of the bars
    # # bar(ind, net(input[digitpat]), width, color='b') #make a plot
    # # xticks(ind+width/2., ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0'))
    # # xlim(-width,N-width)
    # # axhline(linewidth=1, color='black')
    # # title("Trained network (64-10-10-10) guesses a digit above...")
    # # xlabel("Digit")
    # # ylabel("Network outputs")

    # # show()
# # except ImportError, e:
    # # print "Cannot make plots. For plotting install matplotlib.\n%s" % e