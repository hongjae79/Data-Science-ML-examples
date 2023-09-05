#Let's import libraries

import numpy as np
from numpy import zeros
from numpy import ones
import matplotlib.pyplot as plt

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

#Let's build ANN.  It is a simple ANN model with N=1500 number of data, D=2 input dimension, M=4 hidden layer dimension, K=3 number of categories

Xtrain, Ytrain, Xtest, Ytest, Ytrain_ind, Ytest_ind = create_data(2) 

N, D = Xtrain.shape
epoch = 15
batch_sz = 32
n_batches = N // batch_sz

M = 4
K = 3

def sigmoid(p):
    sig = 1/(1+np.exp(-p))
    return sig

def softmax(A):
    for i in range(len(A)):
        A[i] = np.exp(A[i])/sum(np.exp(A[i]))
    return A

W = np.random.randn(D,M)
b = np.random.randn(M)
V = np.random.randn(M,K)
c = np.random.randn(K)

lr = 0.01

train_cost_list = []
train_accuracy_list = []
test_cost_list = []
test_accuracy_list = []

for i in range(epoch):

    for j in range(n_batches):
        Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz)]
        Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz)]

        X = Xbatch
        T = Ybatch

        Z = sigmoid(X.dot(W)+b)
        yhat = softmax(Z.dot(V)+c)

        predict_train = softmax(sigmoid(Xtrain.dot(W)+b).dot(V)+c)
        predict_train_hat = np.argmax(predict_train, axis = 1)

        train_cost = -(Ytrain_ind * np.log(predict_train)).sum()
        train_cost = train_cost / len(Xtrain)
        train_error = error_rate(Ytrain, predict_train_hat)
        train_accuracy = 1 - train_error
        
        predict_test = softmax(sigmoid(Xtest.dot(W)+b).dot(V)+c)
        predict_test_hat = np.argmax(predict_test, axis = 1)
        
        test_cost = -(Ytest_ind * np.log(predict_test)).sum()
        test_cost = test_cost / len(Xtest)
        test_error = error_rate(Ytest, predict_test_hat)
        test_accuracy = 1 - test_error

        JW = -X.T.dot((T-yhat).dot(V.T)*Z*(1-Z))
        Jb = -((T-yhat).dot(V.T)*Z*(1-Z)).sum(axis = 0)
        JV = -Z.T.dot(T - yhat)
        Jc = -(T-yhat).sum(axis = 0)

        W = W - lr * JW
        b = b - lr * Jb
        V = V - lr * JV
        c = c - lr * Jc  

        train_cost_list.append(train_cost) 
        train_accuracy_list.append(train_accuracy)
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

