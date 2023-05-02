#!/usr/bin/python3

import torch
import torch.nn as nn

PATH = "/home/belal/git/pytorch_tutorial/models/model_2.pth"

class Model(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features=6)
# tran your model...

#=======================#
## Save Complete Model ##
#=======================#
# torch.save(model, PATH)

# # model class should be defined somewhere
# model = torch.load(PATH)
# model.eval()

# for param in model.parameters():
#     print(param)

#================#
# ## STATE DICT ##
#================#
# torch.save(model.state_dict(), PATH)

# # model must be created again with parameters
# # model = Model(*args, **kwargs)

# model.load_state_dict(torch.load(PATH))
# model.eval()

# for param in model.parameters():
#     print(param)

#=================#
# ## Check POint ##
#=================#
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}

FILENAME = "checkpoint.pth"

torch.save(checkpoint, FILENAME)

loaded_checkpoint = torch.load(FILENAME)
epoch = loaded_checkpoint["epoch"]

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)

model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optim_state"])

print(optimizer.state_dict())