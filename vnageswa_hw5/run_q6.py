# -*- coding: utf-8 -*-
"""run_q6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xIgOVZCsgwtpxFjf-pCQg6YyWiE_EaLi
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms

import scipy.io

import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class FullyConnectedNet(nn.Module):
    def __init__(self):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(1024, 64)  # Adjust input size and hidden layers as needed
        self.fc2 = nn.Linear(64, 36)   # Adjust output size based on number of classes

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.tensor(x).float().to(device)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        out = torch.nn.functional.softmax(x, dim=1)

        return out

def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################

    N = x.shape[0]
    num_batches = N//batch_size

    shuffled_inds = np.arange(x.shape[0])
    np.random.shuffle(shuffled_inds)

    x_shuffled = x[shuffled_inds]
    y_shuffled = y[shuffled_inds]


    for i in range(num_batches):
        x_batch = x_shuffled[i*batch_size: (i+1)*batch_size,:]
        y_batch = y_shuffled[i*batch_size: (i+1)*batch_size,:]
        batches.append((x_batch, y_batch))


    return batches

def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    ##########################

    loss = -np.multiply(y,np.log(probs)).sum()

    # obtain y_pred from the probs
    y_pred = np.zeros_like(probs)
    max_indices = np.argmax(probs, axis=1)
    y_pred[np.arange(probs.shape[0]), max_indices] = 1

    acc = np.sum(np.multiply(y,y_pred))/y.shape[0]

    return loss, acc

# load data

train_data = scipy.io.loadmat('/content/drive/MyDrive/SEM - 1/16-720 CV/Assignments/hw5/data/nist36_train.mat')
valid_data = scipy.io.loadmat('/content/drive/MyDrive/SEM - 1/16-720 CV/Assignments/hw5/data/nist36_valid.mat')
test_data = scipy.io.loadmat('/content/drive/MyDrive/SEM - 1/16-720 CV/Assignments/hw5/data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

# configurations
max_iters = 200
batch_size = 50

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

train_error = []
train_acc = []

input_size = train_x.shape[1]
output_size = train_y.shape[1]
hidden_size = 64

layers = [input_size, hidden_size, hidden_size, output_size]

model = FullyConnectedNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0

    for xb,yb in batches:
        xb = torch.tensor(xb).float().to(device)
        yb = torch.tensor(yb).float().to(device)

        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()

        out_cpu = output.detach().cpu().numpy()
        yb_cpu = yb.detach().cpu().numpy()

        l, a = compute_loss_and_acc(yb_cpu, out_cpu)

        total_loss += loss.item()
        avg_acc += a

    avg_acc /= batch_num

    train_error.append(total_loss)
    train_acc.append(avg_acc)

    print(f"itr: {itr} || loss :{total_loss}")

# confusion matrix
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

xb = torch.tensor(test_x).float().to(device)
probs = model(xb).detach().cpu().numpy()

# obtain y_pred from the probs
test_y_pred = np.zeros_like(probs)
max_indices = np.argmax(probs, axis=1)
# true_max_indices = np.argmax(test_y, axis=1)
test_y_pred[np.arange(probs.shape[0]), max_indices] = 1

for ind in range(test_y_pred.shape[0]):
    y_pred_ind = np.argmax(test_y_pred[ind,:])
    y_true_ind = np.argmax(test_y[ind,:])

    confusion_matrix[y_pred_ind, y_true_ind] += 1

import matplotlib.pyplot as plt

epoch_history = [i for i in range(len(train_error))]

fig, ax = plt.subplots()
ax.plot(epoch_history, train_error, label='train acc', color='blue')

ax.set_xlabel('epochs')
ax.set_ylabel('loss')
ax.legend()

plt.show()

epoch_history = [i for i in range(len(train_acc))]

fig, ax = plt.subplots()
ax.plot(epoch_history, train_acc, label='train acc', color='blue')

ax.set_xlabel('epochs')
ax.set_ylabel('acc')
ax.legend()

plt.show()

print(f"valid accuracy: {confusion_matrix.trace()/confusion_matrix.sum()}")

class CNN(nn.Module):
    def __init__(self, num_classes=36):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 256, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 36)


    def forward(self, x):

        # x = self.pool(F.relu(self.conv1(x)))
        # print(f"layer 1: {x.shape}")
        # x = self.pool(F.relu(self.conv2(x)))
        # print(f"layer 2: {x.shape}")
        # x = self.pool(F.relu(self.conv3(x)))
        # print(f"layer 2: {x.shape}")
        # x = x.view(-1, 64 * 4 * 4)  # Flatten the tensor for the fully connected layer
        # print(f"layer 3: {x.shape}")
        # x = F.relu(self.fc1(x))
        # print(f"layer 4: {x.shape}")
        # x = self.fc2(x)
        # print(f"layer 5: {x.shape}")


        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        probs = torch.nn.functional.softmax(x, dim=1)

        return x

# load data
# load data
train_data = scipy.io.loadmat('/content/drive/MyDrive/SEM - 1/16-720 CV/Assignments/hw5/data/nist36_train.mat')
valid_data = scipy.io.loadmat('/content/drive/MyDrive/SEM - 1/16-720 CV/Assignments/hw5/data/nist36_valid.mat')
test_data = scipy.io.loadmat('/content/drive/MyDrive/SEM - 1/16-720 CV/Assignments/hw5/data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

# configurations
max_iters = 20
batch_size = 16
learning_rate = 1e-3
hidden_size = 64

# convert the data into images and prepare dataloader

train_x = train_x.reshape(-1, 32, 32)
train_x = np.expand_dims(train_x, axis=1)

valid_x = valid_x.reshape(-1, 32, 32)
valid_x = np.expand_dims(valid_x, axis=1)

train_x_tensor = torch.from_numpy(train_x).float().to(device)
train_y_tensor = torch.from_numpy(train_y).float().to(device)
valid_x_tensor = torch.from_numpy(valid_x).float().to(device)
valid_y_tensor = torch.from_numpy(valid_y).float().to(device)


train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor),
                                           batch_size=batch_size,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(valid_x_tensor, valid_y_tensor),
                                          batch_size=batch_size,
                                          shuffle=True)

batch_num = len(train_loader)

model = CNN().to(device)

train_loss_history = []
valid_loss_history = []
train_acc_history = []
valid_acc_history = []

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# train the model

model.train()

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0

    for xb,yb in train_loader:
        xb = torch.tensor(xb).float().to(device)
        yb = torch.tensor(yb).float().to(device)

        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()

        out_cpu = output.detach().cpu().numpy()
        yb_cpu = yb.detach().cpu().numpy()

        l, a = compute_loss_and_acc(yb_cpu, out_cpu)

        total_loss += loss.item()
        avg_acc += a

    avg_acc /= batch_num

    train_loss_history.append(total_loss)
    train_acc_history.append(avg_acc)

    print(f"itr: {itr} || loss :{total_loss}")

import matplotlib.pyplot as plt

epoch_history = [i for i in range(len(train_loss_history))]

fig, ax = plt.subplots()
ax.plot(epoch_history, train_loss_history, label='train loss', color='blue')

ax.set_xlabel('epochs')
ax.set_ylabel('loss')
ax.legend()

plt.show()

epoch_history = [i for i in range(len(train_acc_history))]

fig, ax = plt.subplots()
ax.plot(epoch_history, train_acc_history, label='train acc', color='blue')

ax.set_xlabel('epochs')
ax.set_ylabel('acc')
ax.legend()

plt.show()

model.eval()

output = model(valid_x_tensor)
loss = criterion(output, valid_y_tensor)

out_cpu = output.detach().cpu().numpy()
yb_cpu = valid_y_tensor.detach().cpu().numpy()

l, a = compute_loss_and_acc(yb_cpu, out_cpu)

print(a)