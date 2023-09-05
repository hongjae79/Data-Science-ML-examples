import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
M = 3
K = 3
epoch = 15
batch_sz = 32
n_batches = N // batch_sz
lr = 0.01

Xtrain = torch.from_numpy(Xtrain).float()
Ytrain = torch.from_numpy(Ytrain).long()
Xtest = torch.from_numpy(Xtest).float()
Ytest = torch.from_numpy(Ytest).long()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(D, M, K).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_cost_list = []
train_accuracy_list = []
test_cost_list = []
test_accuracy_list = []


for i in range(epoch):
    for j in range(n_batches):
        Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz)]
        Ybatch = Ytrain[j*batch_sz:(j*batch_sz + batch_sz)]

        Xbatch = Xbatch.reshape(-1, 2).to(device)
        Ybatch = Ybatch.to(device)

        outputs = model(Xbatch)
        loss = loss_function(outputs, Ybatch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_output = model(Xtrain)
            _, train_prediction = torch.max(train_output.data, 1)
            train_error = error_rate(train_prediction.data.numpy(), Ytrain.data.numpy())
            train_accuracy = 1- train_error
            train_loss = loss_function(train_output, Ytrain)
            train_loss = train_loss.item()

            test_output = model(Xtest)
            _, test_prediction = torch.max(test_output.data, 1)
            test_error = error_rate(test_prediction.data.numpy(), Ytest.data.numpy())
            test_accuracy = 1- test_error
            test_loss = loss_function(test_output, Ytest)
            test_loss = test_loss.item()

            train_cost_list.append(train_loss) 
            train_accuracy_list.append(train_accuracy)
            test_cost_list.append(test_loss) 
            test_accuracy_list.append(test_accuracy)

            if j % 10 == 0:
                print("At iteration, i = %d:, j = %d, loss: %.4f, accuracy: %.4f, val_loss: %.4f, val_accuracy: %.4f" % (i, j, train_loss, train_accuracy, test_loss, test_accuracy))
    
plt.plot(train_cost_list, label='loss')
plt.plot(test_cost_list, label='val_loss')
plt.legend()
plt.show()

plt.plot(train_accuracy_list, label='accuracy')
plt.plot(test_accuracy_list, label='val_accuracy')
plt.legend()
plt.show()