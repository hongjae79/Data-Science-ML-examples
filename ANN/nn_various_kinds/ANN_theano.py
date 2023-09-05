#Let's import libraries

import numpy as np
from numpy import zeros
from numpy import ones
import matplotlib.pyplot as plt
import theano
import theano.tensor as T 

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 3))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def error_rate(p,t):
    return np.mean(p != t)

def create_data(D):

    X1 = np.random.randn(500,D) + np.array([0,-4])
    X2 = np.random.randn(500,D) + np.array([4,-4])
    X3 = np.random.randn(500,D) + np.array([2,0])
    Xtrain = np.concatenate((X1[:-100],X2[:-100],X3[:-100]), axis = 0)
    Xtest = np.concatenate((X1[-100:],X2[-100:],X3[-100:]), axis = 0)

    y1 = np.zeros(500)
    y2 = np.ones(500)
    y3 = 2 * np.ones(500) 
    Ytrain = np.concatenate((y1[:-100],y2[:-100],y3[:-100]), axis = 0)
    Ytest = np.concatenate((y1[-100:],y2[-100:],y3[-100:]), axis = 0)

    #plt.scatter(X1[:,0], X1[:,1], color = 'red')
    #plt.scatter(X2[:,0], X2[:,1], color = 'blue')
    #plt.scatter(X3[:,0], X3[:,1], color = 'yellow')
    #plt.show()

    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    Xtrain = Xtrain.astype(np.float32)
    Ytrain = Ytrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)
    Ytest = Ytest.astype(np.float32)
    Ytrain_ind = y2indicator(Ytrain).astype(np.float32)
    Ytest_ind = y2indicator(Ytest).astype(np.float32)

    return Xtrain, Ytrain, Xtest, Ytest, Ytrain_ind, Ytest_ind

lr = 0.01
#reg = 0.01 

Xtrain, Ytrain, Xtest, Ytest, Ytrain_ind, Ytest_ind = create_data(2)

N, D = Xtrain.shape
epoch = 15
batch_sz = 32
n_batches = N // batch_sz

M = 4
K = 3

W1_init = np.random.randn(D,M)/np.sqrt(D)
b1_init = np.zeros(M)
W2_init = np.random.randn(M,K)/np.sqrt(M)
b2_init = np.zeros(K)

thX = T.matrix('X')
thT = T.matrix('T')
W1 = theano.shared(W1_init, 'W1')
b1 = theano.shared(b1_init, 'b1')
W2 = theano.shared(W2_init, 'W2')
b2 = theano.shared(b2_init, 'b2')

thZ = T.nnet.sigmoid(thX.dot(W1) + b1)
thY = T.nnet.softmax(thZ.dot(W2) + b2)

cost = -(thT * T.log(thY)).sum() #+ reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())
prediction = T.argmax(thY, axis = 1)

update_W1 = W1 - lr*T.grad(cost, W1)
update_b1 = b1 - lr*T.grad(cost, b1)
update_W2 = W2 - lr*T.grad(cost, W2)
update_b2 = b2 - lr*T.grad(cost, b2)

train = theano.function(
    inputs = [thX, thT], 
    updates = [(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)],
    )

get_prediction = theano.function(
    inputs = [thX, thT], 
    outputs = [cost, prediction],
)

train_cost_list = []
train_accuracy_list = []
test_cost_list = []
test_accuracy_list = []

for i in range(epoch):
    for j in range(n_batches):
    
        train(Xtrain, Ytrain_ind)

        train_cost, train_prediction = get_prediction(Xtrain, Ytrain_ind)
        train_err = error_rate(train_prediction, Ytrain)
        train_cost = train_cost / len(Xtrain)
        train_accuracy = 1 - train_err
        train_cost_list.append(train_cost)
        train_accuracy_list.append(train_accuracy)

        test_cost, test_prediction = get_prediction(Xtest, Ytest_ind)
        test_err = error_rate(test_prediction, Ytest)
        test_cost = test_cost / len(Xtest)
        test_accuracy = 1 - test_err
        test_cost_list.append(test_cost)
        test_accuracy_list.append(test_accuracy)

        if j % 10 == 0:
            print("At iteration, i = %d:, j = %d, loss: %.4f, accuracy: %.4f, val_loss: %.4f, val_accuracy: %.4f" % (i, j, train_cost, train_accuracy, test_cost, test_accuracy))



plt.plot(train_cost_list, label='loss')
plt.plot(test_cost_list, label='val_loss')
plt.legend()
plt.show()

plt.plot(train_accuracy_list, label='accuracy')
plt.plot(test_accuracy_list, label='val_accuracy')
plt.legend()
plt.show()