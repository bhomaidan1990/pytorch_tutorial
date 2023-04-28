#!/usr/bin/python3

import torch

if torch.cuda.is_available():
    device_ = torch.device("cuda")
    print("========================\nYou are running on GPU!\n========================")
else:
    device_ = torch.device("cpu")
    print("------------------------\nYou are running on CPU!\n------------------------")

# Linear Regression f = w * x (ignore bias)
# Y = f(X) = 2 * X
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device=device_)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32, device=device_)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device_)

# Model Prediction
def forward(x):
    return w * x

# Loss MSE
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()

# Gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x (w*x - y)
"""
def gradient(x, y, y_pred):
    return torch.dot(2*x, y_pred - y).mean()
"""

print(f'prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    l.backward()

    # update whights (gradient decent)
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero gradients
    w.grad.zero_()

    if(epoch % 10 == 0):
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')