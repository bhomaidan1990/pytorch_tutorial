#!/usr/bin/python3

import torch
import torch.nn as nn

if torch.cuda.is_available():
    device_ = torch.device("cuda")
    print("========================\nYou are running on GPU!\n========================")
else:
    device_ = torch.device("cpu")
    print("------------------------\nYou are running on CPU!\n------------------------")

# import torch.nn.functional as F
# F.leaky_relu()

# Multiclass problem
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.linear1(x)
        # out = self.relu(out)
        out = torch.relu(out)
        out = self.linear2(out)
        # no softmax; return out
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred
    
model     = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
# criterion = nn.CrossEntropyLoss() # (applies Softmax)
criterion = nn.BCELoss()