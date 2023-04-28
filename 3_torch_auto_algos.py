#!/usr/bin/python3

import torch
import torch.nn as nn

if torch.cuda.is_available():
    device_ = torch.device("cuda")
    print("========================\nYou are running on GPU!\n========================")
else:
    device_ = torch.device("cpu")
    print("------------------------\nYou are running on CPU!\n------------------------")

"""
Trining Pipeline:
-----------------
1. Design Model (input size, output size, forward pass)
2. Construct Loss and Optimizer
3. Training Loop
  3.1 Forward pass: compute prediction
  3.2 Backward pass: gradients 
  3.3 Update whights
"""

# Linear Regression f = w * x (ignore bias)
# Y = f(X) = 2 * X
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32, device=device_)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32, device=device_)

X_test = torch.tensor([5], dtype=torch.float32, device=device_)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size)
# model = model.to(device_)


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define Layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
    
model = LinearRegression(input_size, output_size)
model = model.to(device_)
# Loss MSE

# Gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x (w*x - y)
"""
def gradient(x, y, y_pred):
    return torch.dot(2*x, y_pred - y).mean()
"""

print(f'prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.05
n_iters = 500

loss_ = nn.MSELoss()
optimizer_ = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss_(Y, y_pred)

    # gradients
    l.backward()

    # update whights (gradient decent)
    optimizer_.step()

    # zero gradients
    optimizer_.zero_grad()

    if(epoch % 10 == 0):
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')