import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

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

Xtrain, Ytrain, Xtest, Ytest, Ytrain_ind, Ytest_ind = create_data(2)

N, D = Xtrain.shape
epoch = 15
batch_sz = 32
n_batches = N // batch_sz

M1 = 4
M2 = 4
K = 3

W1_init = np.random.randn(D,M1)/np.sqrt(D)
b1_init = np.zeros(M1)
W2_init = np.random.randn(M1,M2)/np.sqrt(M1)
b2_init = np.zeros(M2)
W3_init = np.random.randn(M2,K)/np.sqrt(M2)
b3_init = np.zeros(K)

X = tf.compat.v1.placeholder(tf.float32, shape = (None, D), name = 'X')
T = tf.compat.v1.placeholder(tf.float32, shape = (None, K), name = 'T')
W1 = tf.Variable(W1_init.astype(np.float32))
b1 = tf.Variable(b1_init.astype(np.float32))
W2 = tf.Variable(W2_init.astype(np.float32))
b2 = tf.Variable(b2_init.astype(np.float32))
W3 = tf.Variable(W3_init.astype(np.float32))
b3 = tf.Variable(b3_init.astype(np.float32))

Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
Yish = tf.matmul(Z2, W3) + b3

cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = Yish, labels = T))
#softmax_cross_entropy_with_logits
lr = 0.01

train_op = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost)

predict_op = tf.argmax(Yish, 1)

init = tf.compat.v1.initialize_all_variables()

train_cost_list = []
train_accuracy_list = []
test_cost_list = []
test_accuracy_list = []

with tf.compat.v1.Session() as session:
    session.run(init)

    for i in range(epoch):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
            
            session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})

            train_cost = session.run(cost, feed_dict={X: Xtrain, T: Ytrain_ind})
            train_prediction = session.run(predict_op, feed_dict={X:Xtrain})
            train_cost = train_cost / len(Xtrain)
            train_err = error_rate(train_prediction, Ytrain)
            train_accuracy = 1 - train_err
            train_cost_list.append(train_cost)
            train_accuracy_list.append(train_accuracy)
            
            test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest_ind})
            test_prediction = session.run(predict_op, feed_dict={X:Xtest})
            test_cost = test_cost/len(Xtest)
            test_err = error_rate(test_prediction, Ytest)
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

