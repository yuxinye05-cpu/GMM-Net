import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchstat import stat


class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()

        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 基于1*1卷积学习通道之间的信息
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        # 激活函数
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1]
        v = self.avg_pool(x)

        # 然后，基于1*1卷积学习通道之间的信息；其中，使用前面设计的自适应卷积核
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # 最终，经过sigmoid 激活函数处理
        v = self.sigmoid(v)
        # return x * v * self.gamma + x
        return x * v + x



class MultiScaleInputModule(nn.Module):
    def __init__(self, output_channels,num_floors,input_channels=1):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_floors = num_floors

        # 定义不同尺度的处理分支
        self.pool1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(input_channels, output_channels, 3,1,1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.GELU(),
                                   nn.Dropout2d(0.1)
                                   )
        self.pool2 = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4),
                                   nn.Conv2d(input_channels, output_channels, 3,1,1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.GELU(),
                                   nn.Dropout2d(0.2)
                                   )
        self.pool3 = nn.Sequential(nn.MaxPool2d(kernel_size=8, stride=8),
                                   nn.Conv2d(input_channels, output_channels, 3,1,1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.GELU(),
                                   nn.Dropout2d(0.3)
                                   )

        self.eca= ECA(2*output_channels)
        self.conv1x1 = nn.Sequential(nn.Conv2d(output_channels, 2*output_channels, kernel_size=1, stride=1),
                                     nn.BatchNorm2d(2*output_channels),
                                     nn.GELU(),
                                     nn.Dropout2d(0.1)
                                     )
        self.gamma = nn.Parameter(torch.zeros(1))





    def forward(self, x1,x2):
        if self.num_floors==1:
            x1=self.pool1(x1)
        if self.num_floors==2:
            x1=self.pool2(x1)
        if self.num_floors==3:
            x1=self.pool3(x1)
        x1=torch.cat([x1,x2],dim=1)
        x1=self.eca(x1)
        x2=self.conv1x1(x2)
        return self.gamma*x1+x2

if __name__ == '__main__':
    x1 = torch.randn(4, 1, 128, 128)
    x2= torch.randn(4, 128, 16, 16)
    model = MultiScaleInputModule(128,3)
    out = model(x1,x2)
    print('out.shape:{}'.format(out.shape))