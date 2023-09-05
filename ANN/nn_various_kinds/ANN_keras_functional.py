from keras.models import Model
from keras.layers import Dense, Input

import matplotlib.pyplot as plt
import numpy as np

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
K = 3

# ANN with [2] -> [4] -> [3]
i = Input(shape=(D,))
x = Dense(4, activation='sigmoid')(i)
x = Dense(K, activation='softmax')(x)
model = Model(inputs = i, outputs = x)

model.compile(
    loss = 'categorical_crossentropy', 
    optimizer = 'adam', 
    metrics=['accuracy']
)

r = model.fit(Xtrain, Ytrain_ind, validation_data=(Xtest, Ytest_ind), epochs = 15, batch_size = 32)

print('Returend:', r)
print(r.history.keys())
print(N)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()