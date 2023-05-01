#!/usr/bin/python3

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device_ = torch.device("cuda")
    print("========================\nYou are running on GPU!\n========================")
else:
    device_ = torch.device("cpu")
    print("------------------------\nYou are running on CPU!\n------------------------")

"""
0) prepare data
1) model
2) loss and optimizer
3) training loop
  3.1 Forward pass: compute prediction
  3.2 Backward pass: gradients 
  3.3 Update whights
"""

# 0) Data Perparation
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0], 1)

# to GPU
X = X.to(device_)
Y = Y.to(device_)

# 1) Model
n_samples, n_features = X.shape
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size, device=device_)

# 2) Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass
    y_pred = model(X)
    loss = criterion(y_pred, Y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    optimizer.zero_grad()

    if((epoch+1) % 10==0):
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# plot
predicted = model(X).detach().cpu().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()