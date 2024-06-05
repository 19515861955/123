#方法一
import torch.nn as nn
import torch.nn.functional as F

#定义卷积神经网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层 (32x32x3的图像)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 卷积层(16x16x16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 卷积层(8x8x32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 100)
        # dropout层 (p=0.3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x

#方法二
import torch
class MLP(torch.nn.Module):

    def __init__(self, num_i, num_h, num_o):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


#方法三
from torch import nn
# 定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 定义残差块里连续的2个卷积层
        self.block_conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride!=1的时候 也就是每次输出信道升维的时候
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out1 = self.block_conv(x)
        out2 = self.shortcut(x) + out1
        out2 = F.relu(out2)  # F.relu()是函数调用，一般使用在foreward函数里。而nn.ReLU()是模块调用，一般在定义网络层的时候使用
        return out2


# 构建RESNET18
class ResNet_18(nn.Module):
    def __init__(self, ResBlock, num_classes):
        super(ResNet_18, self).__init__()
        self.in_channels = 64  # 输入layer1时的channel
        # 第一层单独卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0)  # 64
        )

        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)  # 64
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)  # 32
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)  # 16
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)  # 8

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # torch.nn.AdaptiveAvgPool2d()接受两个参数，分别为输出特征图的长和宽，其通道数前后不发生变化。
        # 即这里将输入图片像素强制转换为1*1
        self.linear = nn.Linear(512 * 1 * 1, num_classes)

        self.dropout = nn.Dropout(0.3)

    # 这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.dropout(x)
        return x
