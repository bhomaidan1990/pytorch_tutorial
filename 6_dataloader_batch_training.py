#!/usr/bin/python3

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

if torch.cuda.is_available():
    device_ = torch.device("cuda")
    print("========================\nYou are running on GPU!\n========================")
else:
    device_ = torch.device("cpu")
    print("------------------------\nYou are running on CPU!\n------------------------")

class WineDataset(Dataset):
    
    def __init__(self):
        # data loading
        xy = np.loadtxt("./dataset/wine.csv", dtype=np.float32, delimiter=",",  skiprows=1)
        self.n_samples = xy.shape[0]

        self.x = torch.from_numpy(xy[:, 1:])
        # self.x = self.x.to(device_)

        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        # self.y = self.y.to(device_)

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples

dataset = WineDataset()
train_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward backward, update
        if((i+1) % 5 ==0):
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')