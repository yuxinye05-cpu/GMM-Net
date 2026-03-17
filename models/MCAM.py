import math
import torch
from torchstat import stat
from torch import nn


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()

        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 基于1*1卷积学习通道之间的信息
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1]
        v = self.avg_pool(x)

        # 然后，基于1*1卷积学习通道之间的信息；其中，使用前面设计的自适应卷积核
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # 最终，经过sigmoid 激活函数处理
        v = self.sigmoid(v)
        return v


# 定义激活函数h_sigmoid
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


# 定义激活函数h_swish
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


# 位置注意力模块
class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # 使用自适应池化进行垂直池化
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 进行水平池化，行列转置
        y = torch.cat([x_h, x_w], dim=2)  # 按行拼接
        y = self.conv1(y)  # 改变通道数
        y = self.bn1(y)
        y = self.act(y)  # 使用h_swish激活函数
        x_h, x_w = torch.split(y, [h, w], dim=2)  # 按行拆分
        x_w = x_w.permute(0, 1, 3, 2)  # 行列转置
        a_h = self.conv_h(x_h).sigmoid()  # 改变通道数
        a_w = self.conv_w(x_w).sigmoid()
        out = a_w * a_h
        return out


class MCAM(nn.Module):

    def __init__(self, channels=64):
        super(MCAM, self).__init__()
        # 局部注意力
        self.cda = CoordAtt(channels)

        # 全局注意力
        self.eca = ECABlock(channels)

        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        xl = self.cda(x)
        xg = self.eca(x)
        xlg = xl + xg
        # xlg = xl * xg
        att = self.sigmoid(xlg)
        # out = x * att * self.gamma + x
        out=x*att+x
        return out


if __name__ == '__main__':
    img = torch.randn(32, 256, 32, 32)
    model = MCAM(256)
    out = model(img)
    print('out.shape:{}'.format(out.shape))
    stat(model, (256, 64, 64))
