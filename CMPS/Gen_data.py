import numpy as np
import torch
from torchvision import datasets,transforms
data_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
train_dataset=datasets.MNIST(root='./train_data',train=True,transform=data_tf,download=True)
test_dataset=datasets.MNIST(root='./test_data',train=False,transform=data_tf,download=True)
print(len(train_dataset))
train_data = []
train_label = []
test_data = []
test_label = []
for i in range(len(train_dataset)):
    train_data.append(train_dataset[i][0].numpy())
    train_label.append(train_dataset[i][1])
for i in range(len(test_dataset)):
    test_data.append(test_dataset[i][0].numpy())
    test_label.append(test_dataset[i][1])
train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)
print(train_data.shape)
print(train_data.dtype)
print(test_data.shape)
print(test_data.dtype)

train_data = np.where(train_data>0,1,0)
test_data = np.where(test_data>0,1,0)
train_data = train_data.reshape(train_data.shape[0],28*28)
test_data = test_data.reshape(test_data.shape[0],28*28)
print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)
print(train_data.max(),train_data.min())
np.savez('data_bin.npz', train_data = train_data, train_label = train_label, test_data = test_data, test_label = test_label)

