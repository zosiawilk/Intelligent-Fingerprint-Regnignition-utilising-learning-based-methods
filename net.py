import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)
    
class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)


def Conv_Block(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
        nn.BatchNorm2d(out_channel),
        nn.Dropout2d(0.3),
        nn.LeakyReLU(),
        nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
        nn.BatchNorm2d(out_channel),
        nn.Dropout2d(0.3),
        nn.LeakyReLU(),
    )

class BasicBlock(nn.Module):
    expansion = 1  # 将expansion设置为1，保持通道数不变
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:  # 修改连接处的通道数
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        return nn.LeakyReLU()(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, stride=stride, kernel_size=3,padding_mode= 'reflect', padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel * BottleNeck.expansion, kernel_size=1,padding_mode= 'reflect', bias=False),
            nn.BatchNorm2d(out_channel * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channel != out_channel * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.LeakyReLU()(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, in_channel, out_channel, block,num):
        super().__init__()

        self.in_channel = 64*block.expansion

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channel, 64*block.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64*block.expansion),
            nn.Dropout2d(0.3),
            nn.LeakyReLU())
        self.d1 = DownSample(64*block.expansion)
        self.c2 = self._make_layer(block, 128, num[0], 1) #type of block,Out_channels,num_blocks[3,4,6,3],stride
        self.d2 = DownSample(128*block.expansion)
        self.c3 = self._make_layer(block, 256, num[1], 1)
        self.d3 = DownSample(256*block.expansion)
        self.c4 = self._make_layer(block, 512, num[2], 1)
        self.d4 = DownSample(512*block.expansion)
        self.c5 = self._make_layer(block, 1024,num[3], 1)
        self.u1=UpSample(1024*block.expansion)
        self.c6 = self._make_layer(block, 512,num[3], 1)
        self.u2 = UpSample(512*block.expansion)
        self.c7 = self._make_layer(block, 256,num[2], 1)
        self.u3 = UpSample(256*block.expansion)
        self.c8 = self._make_layer(block,128,num[1], 1)
        self.u4 = UpSample(128*block.expansion)
        self.c9 = self._make_layer(block,64,num[0],1)
        self.out=nn.Conv2d(64*block.expansion,out_channel,3,1,1)
        
    def _make_layer(self, block, out_channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        R1=self.c1(x)
        R2=self.c2(self.d1(R1))

        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1=self.c6(self.u1(R5,R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))
        return self.out(O4)

def resnet_e(in_channel, out_channel):
    model = ResNet(in_channel, out_channel, BasicBlock,[3,4,6,3])
    return model

def resnet_h(in_channel, out_channel):
    model = ResNet(in_channel, out_channel, BottleNeck,[3,4,6,3])
    return model

if __name__ == '__main__':
    net = resnet_h(1,1)
    x = torch.rand((1, 1, 128, 128))
    print(net.forward(x).shape)
