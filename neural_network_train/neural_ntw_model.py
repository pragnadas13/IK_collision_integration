import torch
import torch.nn as nn

class DetectCost(nn.Module):
    def __init__(self):
        super(DetectCost, self).__init__()
        self.fc1 = nn.Linear(10, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6,1)
        self.relu1 = nn.ReLU() # instead of Heaviside step fn
        self.relu2 = nn.ReLU()

    def forward(self, x, y):
        print(x.shape)
        print(y.shape)
        # if len(x.shape) ==1:
        #     output = torch.cat((x, y), 1)
        output = torch.cat((x, y), 1)#this function have to be called to train , what is the input?
        print(output.shape)
        output = self.fc1(output)
        # print("FC1")
        # print(output.shape)
        output = self.relu1(output) # instead of Heaviside step fn
        output = self.fc2(output)
        # print("FC2")
        # print(output.shape)
        output = self.relu2(output)
        output = self.fc3(output)
        # print('output')
        # print(output.shape)
        return output