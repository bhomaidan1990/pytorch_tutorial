#!/usr/bin/python3

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import tqdm

if torch.cuda.is_available():
    device_ = torch.device("cuda")
    print("========================\nYou are running on GPU!\n========================")
else:
    device_ = torch.device("cpu")
    print("------------------------\nYou are running on CPU!\n------------------------")

# hyper parameters
input_size = 784 # 28 x 28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size=100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transforms.ToTensor(), 
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)
# print(samples.shape, labels.shape)

# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(samples[i][0], cmap="gray")
# plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device_)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    loop = tqdm(train_loader)
    for i, (images, labels) in enumerate(loop):
        # 100, 1, 28, 28
        # 100, 784
        images = images.reshape(-1, 28*28).to(device_)
        labels = labels.to(device_)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

         # # add stuff to progress bar in the end
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())

        # if((i+1) % 100 == 0):
        #     print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
        
# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device_)
        labels = labels.to(device_)
        outputs = model(images)
        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct = (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc*100} %')
