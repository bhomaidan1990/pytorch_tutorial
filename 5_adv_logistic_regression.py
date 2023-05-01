#!/usr/bin/python3

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
bc = datasets.load_breast_cancer()

X = bc.data
Y = bc.target
n_samples, n_features = X.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
X_test  = torch.from_numpy(X_test.astype(np.float32))
Y_test  = torch.from_numpy(Y_test.astype(np.float32))

Y_train = Y_train.view(Y_train.shape[0], 1)
Y_test  = Y_test.view(Y_test.shape[0], 1)

# to GPU
X_train = X_train.to(device_)
X_test  = X_test.to(device_)
Y_train = Y_train.to(device_)
Y_test  = Y_test.to(device_)

# 1) Model
# f = wx +b, sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1, device=device_)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = LogisticRegression(n_features)



# 2) Loss and Optimizer
learning_rate = 0.05
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 400
for epoch in range(num_epochs):
    # forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, Y_train)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    optimizer.zero_grad()

    if((epoch+1) % 10==0):
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# Evaluation
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
