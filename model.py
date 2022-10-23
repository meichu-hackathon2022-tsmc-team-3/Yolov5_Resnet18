import torch
import torch.nn as nn
from torchvision import models


class ResNet152(nn.Module):

    def __init__(self):

        super(ResNet152, self).__init__()
        self.resnet152 = models.resnet152()
        # for param in self.resnet152.parameters():
        #     param.require_grad = False
        self.resnet152.fc = nn.Linear(2048, 2)
        
    def forward(self, x):
        x = self.resnet152(x)
        return x


class ResNet18(nn.Module):

    def __init__(self):

        super(ResNet18, self).__init__()
        self.resnet152 = models.resnet18()
        # for param in self.resnet152.parameters():
        #     param.require_grad = False
        self.resnet152.fc = nn.Linear(512, 2)
        
    def forward(self, x):
        x = self.resnet152(x)
        return x


class ResNet34(nn.Module):

    def __init__(self):

        super(ResNet34, self).__init__()
        self.resnet34 = models.resnet34()
        # for param in self.resnet152.parameters():
        #     param.require_grad = False
        self.resnet34.fc = nn.Linear(512, 2)
        
    def forward(self, x):
        x = self.resnet34(x)
        return x


if __name__ == '__main__':

    model = ResNet18()
    print(model)