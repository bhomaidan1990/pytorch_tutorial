#!/usr/bin/python3

import torch
import torch.nn as nn
import numpy as np

if torch.cuda.is_available():
    device_ = torch.device("cuda")
    print("========================\nYou are running on GPU!\n========================")
else:
    device_ = torch.device("cpu")
    print("------------------------\nYou are running on CPU!\n------------------------")

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
# print('softmax numpy:', outputs)

x = torch.tensor([2.0, 1.0, 0.1], device=device_)
output = torch.softmax(x, dim=0)
# print(outputs)

# y_pred has probabilities
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad  = np.array([0.1, 0.3, 0.6])

# y must be one hot encoded
Y = np.array([1, 0, 0])

l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)

print(f'Loss numpy: {l1:.4f}')
print(f'Loss numpy: {l2:.4f}')

loss = nn.CrossEntropyLoss()

Y_ = torch.tensor([2, 0, 1], device=device_)
# n_samples x n_classes = 3 x 3
Y_pred_good_ = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]], device=device_)
Y_pred_bad_  = torch.tensor([[2.1, 1.0, 0.3], [0.1, 2.0, 2.1], [0.1, 3.0, 0.1]], device=device_)

l1_ = loss(Y_pred_good_, Y_)
l2_ = loss(Y_pred_bad_ , Y_)

print(l1_.item())
print(l2_.item())

_, predictions1 = torch.max(Y_pred_good_, 1)
_, predictions2 = torch.max(Y_pred_bad_, 1)

print(predictions1, predictions2)