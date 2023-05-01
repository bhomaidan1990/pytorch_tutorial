#!/usr/bin/python3

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

if torch.cuda.is_available():
    device_ = torch.device("cuda")
    print("========================\nYou are running on GPU!\n========================")
else:
    device_ = torch.device("cpu")
    print("------------------------\nYou are running on CPU!\n------------------------")

dataset = torchvision.datasets.MNIST(
    root='./data', 
    transform=torchvision.transforms.ToTensor(),
    download=True
)

class WineDataset(Dataset):
    
    def __init__(self, transform=None):
        self.transform = transform
        # data loading
        xy = np.loadtxt("./dataset/wine.csv", dtype=np.float32, delimiter=",",  skiprows=1)
        self.n_samples = xy.shape[0]

        self.x = xy[:, 1:]

        self.y = xy[:, [0]] # n_samples, 1

    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples
    
class ToTensor():
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform():
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data