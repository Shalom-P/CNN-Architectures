import torch
import torch.nn as nn
from torch import nn, Tensor


class BasicBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,kernel_size:int,stride:int,padding:int,downsample:bool=False):
        super(BasicBlock,self).__init__()  
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,padding=padding,bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.downsample_flag = False
        if downsample:
            self.downsample_flag = downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
                nn.BatchNorm2d(num_features=out_channels,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
    
    def forward(self,x_in):
        x = self.conv1(x_in)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample_flag:
            x_in = self.downsample(x_in)
        x = x_in+x
        return x

class resnet18(nn.Module):
    def __init__(self,num_classes=2):
        super(resnet18,self).__init__()  
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = nn.Sequential(
            BasicBlock(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            BasicBlock(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1,downsample=True),
            BasicBlock(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1,downsample=True),
            BasicBlock(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1,downsample=True),
            BasicBlock(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512,out_features=num_classes)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
if __name__=='__main__':
    
    mod = resnet18(num_classes=8)
    inn = torch.randn((7,3,224,224))
    out = mod(inn)
    print(out.size())