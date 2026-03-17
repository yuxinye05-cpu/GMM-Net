"""
This part contains UNet series models, 
including UNet, R2UNet, Attention UNet, R2Attention UNet, DenseUNet
"""

# 导入各种模块
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torchvision.models as models
from torch.nn import init
from functools import reduce
from functools import partial
from torch.nn import Softmax
nonlinearity = partial(F.relu, inplace=True)

# ==========================Core Module================================


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
        self.sigmoid = h_sigmoid(inplace=inplace)                   #调用 h_sigmoid激活函数

    def forward(self, x):
        return x * self.sigmoid(x)


# 位置注意力模块
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)                              # 使用自适应池化进行垂直池化
        x_w = self.pool_w(x).permute(0, 1, 3, 2)          # 进行水平池化，且行列转置

        y = torch.cat([x_h, x_w], dim=2)                  # 按行拼接
        y = self.conv1(y)                                 # 改变通道数
        y = self.bn1(y)
        y = self.act(y)                                   # 使用h_swish激活函数

        x_h, x_w = torch.split(y, [h, w], dim=2)          # 按行拆分
        x_w = x_w.permute(0, 1, 3, 2)                     # 行列转置

        a_h = self.conv_h(x_h).sigmoid()                  # 改变通道数
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


# 卷积+BN+relu
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


# 没有调用过
class conv_block6(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block6, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)

        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1)


    def forward(self, x):
        residual = self.conv1x1(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out =out + residual
        out = self.relu(out)

        return out





# 特征编码块：两次3*3卷积+位置注意力+残差连接
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.sa1 = CoordAtt(ch_out, ch_out)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)

        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1)


    def forward(self, x):
        residual = self.conv1x1(x)                                   # 改变通道数
        out = self.relu(self.bn1(self.conv1(x)))                     # 卷积+BN+relu
        out = self.relu(self.bn2(self.sa1(self.conv2(out))))         # 卷积+位置注意力+BN+relu
        out =out + residual                                          # 残差连接
        out = self.relu(out)

        return out


# 上采样+3*3卷积，尺寸扩大两倍
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(                                                          # 把各种模型顺序连接起来
            nn.Upsample(scale_factor=2),                                                  # 使用最邻近算法进行上采样
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


# 1*1卷积，改变通道数
class conv_block1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block1, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1,    bias=True)

    def forward(self, x):
        out = self.conv1(x)
        return out


# 反卷积，尺寸扩大为两倍
class conv_block2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block2, self).__init__()
        self.conv1 = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3,stride=2, padding=1, output_padding=1)  # 反卷积
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return out


# 多尺度特征融合模块
class SAPblock(nn.Module):
    def __init__(self, in_channels):
        super(SAPblock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, dilation=1, kernel_size=3,
                                 padding=1)

        self.bn = nn.ModuleList([nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels)])
        self.conv1x1 = nn.ModuleList(
            [nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0),
             nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0)])
        self.conv3x3_1 = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1),
             nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1)])
        self.conv3x3_2 = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1),
             nn.Conv2d(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1)])
        self.conv_last = ConvBnRelu(in_planes=in_channels, out_planes=in_channels, ksize=1, stride=1, pad=0, dilation=1)
        self.sa1 = CoordAtt(in_channels, in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))                            # 定义残差连接的学习参数

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_size = x.size()

        branches_1 = self.conv3x3(x)
        branches_1 = self.bn[0](branches_1)
        branches_1 = self.sa1(branches_1)                                     # 位置注意力模块
        branches_2 = F.conv2d(x, self.conv3x3.weight, padding=2, dilation=2)  # 空洞卷积，空洞率=2
        branches_2 = self.bn[1](branches_2)
        branches_2 = self.sa1(branches_2)
        branches_3 = F.conv2d(x, self.conv3x3.weight, padding=3, dilation=3)  # 空洞卷积，空洞率=3
        branches_3 = self.bn[2](branches_3)
        branches_3 = self.sa1(branches_3)
        feat = torch.cat([branches_1, branches_2], dim=1)                     # 根据通道拼贴1和2，通道数翻倍
        # feat=feat_cat.detach()
        feat = self.relu(self.conv1x1[0](feat))                               # 通道数减半
        feat = self.relu(self.conv3x3_1[0](feat))                             # 通道数减半
        att = self.conv3x3_2[0](feat)                                         # 通道数变为2
        att = F.softmax(att, dim=1)                                           # 使用softmax函数得到权重

        att_1 = att[:, 0, :, :].unsqueeze(1)                                  # 提取0通道的所有数据（1的权重），并维持维度不变
        att_2 = att[:, 1, :, :].unsqueeze(1)                                  # 提取1通道的所有数据（2的权重），并维持维度不变

        fusion_1_2 = att_1 * branches_1 + att_2 * branches_2                  # 进行1,2的尺度特征融合

        feat1 = torch.cat([fusion_1_2, branches_3], dim=1)                    # 根据通道拼贴1_2和3，通道数翻倍
        # feat=feat_cat.detach()
        feat1 = self.relu(self.conv1x1[0](feat1))                             # 通道数减半
        feat1 = self.relu(self.conv3x3_1[0](feat1))                           # 通道数减半
        att1 = self.conv3x3_2[0](feat1)                                       # 通道数变为2
        att1 = F.softmax(att1, dim=1)                                         # 使用softmax函数的得到权重

        att_1_2 = att1[:, 0, :, :].unsqueeze(1)                               # 提取0通道的所有数据（1_2的权重），并维持维度不变
        att_3 = att1[:, 1, :, :].unsqueeze(1)                                 # 提取1通道的所有数据（3的权重），并维持维度不变
                                                                              # 进行1_2和3的尺度特征融合，并加入残差连接
        ax = self.relu(self.gamma * (att_1_2 * fusion_1_2 + att_3 * branches_3) + (1 - self.gamma) * x)
        ax = self.conv_last(ax)                                               # 进行卷积

        return ax









# ==================================================================
class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2)

        self.Conv1 = conv_block(ch_in=img_ch , ch_out=32)
        self.sa1 = CoordAtt(inp=32, oup=32)
        self.Conv2 = conv_block(ch_in=32, ch_out=48)
        self.sa2 = CoordAtt(inp=48, oup=48)
        self.Conv3 = conv_block(ch_in=48,ch_out=64)
        self.sa3 = CoordAtt(inp=64, oup=64)
        self.Conv4 = conv_block(ch_in=64, ch_out=80)
        self.sa4 = CoordAtt(inp=80, oup=80)

        self.sap = SAPblock(80)


        self.conv5 = conv_block1(ch_in=80, ch_out=48)
        self.conv6 = conv_block1(ch_in=64, ch_out=48)
        self.conv7 = conv_block1(ch_in=80, ch_out=64)


        self.conv11 = conv_block1(ch_in=128, ch_out=64)
        self.conv12 = conv_block1(ch_in=96, ch_out=48)
        self.conv13 = conv_block1(ch_in=64, ch_out=64)
        self.conv14 = conv_block1(ch_in=80, ch_out=80)


        self.conv8 = conv_block2(ch_in=32, ch_out=48)
        self.conv9 = conv_block2(ch_in=48, ch_out=64)
        self.conv10 = conv_block2(ch_in=64, ch_out=80)


        self.Up4 = up_conv(ch_in=80, ch_out=16)
        self.Up_conv4 = conv_block(ch_in=80, ch_out=64)
        self.sa5 = CoordAtt(inp=64, oup=64)

        self.Up3 = up_conv(ch_in=64, ch_out=16)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=48)
        self.sa6 = CoordAtt(inp=48, oup=48)

        self.Up2 = up_conv(ch_in=48, ch_out=16)
        self.Up_conv2 = conv_block(ch_in=48, ch_out=32)
        self.sa7 = CoordAtt(inp=32, oup=32)




        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)                      # 第一层特征编码块             3-32
        #out1 = self.sa1(x1)

        x2 = self.Maxpool(x1)                   # 最大池化后进入编码器第二层     32-32
        p1 = self.conv8(x2)                     # 反卷积，尺寸扩大两倍                      32-48
        x2 = self.Conv2(x2)                     # 第二层特征编码块              32-48
        #out2 = self.sa2(x2)

        x3 = self.Maxpool(x2)                   # 最大池化后进入编码器第三层      48-48
        p2 = self.conv9(x3)                     # 反卷积，尺寸扩大两倍                       48-64
        x3 = self.Conv3(x3)                     # 第三层特征编码块               48-64
        #out3 = self.sa3(x3)

        x4 = self.Maxpool(x3)                   # 最大池化后进入编码器第四层      64-64
        p3 = self.conv10(x4)                    # 反卷积，尺寸扩大两倍                      64-80
        x4 = self.Conv4(x4)                     # 第四层特征编码块                64-80
        #out4 = self.sa4(x4)
        #out4 = self.sap(out4)


        d5 = self.conv5(x4)                     # 1*1卷积        80-48
        d6 = self.conv6(x3)                     # 1*1卷积        64-48
        d7 = self.conv7(x4)                     # 1*1卷积        80-64



        d4 = self.Up4(x4)                       # 上采样+3*3卷积，尺寸扩大两倍          80-16
        d4 = torch.cat((x3, d4), dim=1)         # 拼接        64 + 16 = 80
        d4 = d4+p3                              # 相加            80

        d4 = self.Up_conv4(d4)                  # 第三层特征解码块   80-64
        d7 = self.up(d7)                        # 上采样，尺寸扩大两倍     64
        d7 = self.up(d7)                        # 上采样，尺寸扩大两倍     64

        d3 = self.Up3(d4)                       # 上采样+3*3卷积，尺寸扩大两倍   64-16


        d3 = torch.cat((x2, d3,d7), dim=1)      # 拼接    48+16+64=128
        d3 = self.conv11(d3)                    # 1*1卷积，改变通道数  64
        d3 = d3+p2                              # 相加     64

        d3 = self.Up_conv3(d3)                  # 第二层特征解码块  64-48

        d5 = self.up(d5)                        # 上采样，尺寸扩大8倍   48
        d5 = self.up(d5)
        d5 = self.up(d5)


        d6 = self.up(d6)                        # 上采样，尺寸扩大4倍       48
        d6 = self.up(d6)


        d8 = d6+d5                              # 拼接                   96

        d2 = self.Up2(d3)                       # 上采样+3*3卷积，尺寸扩大两倍  48-16
        d2 = torch.cat((x1, d2, d8), dim=1)     # 拼接                32+16+96=144
        
        d2 = self.conv12(d2)                    # 1*1卷积，改变通道数    96-48
        d2 = d2 +p1                             # 相加               48

        d2 = self.Up_conv2(d2)                  # 第一层特征解码块       48-32


        d1 = self.Conv_1x1(d2)                  # 1*1卷积，改变通道数   32-1
        d1 = F.softmax(d1,dim=1)                # softmax函数

        return d1


# ============================================================
if __name__ == '__main__':            # 所在模块是被直接运行的，则该语句下代码块被运行，如果所在模块是被导入到其他的python脚本中运行的，则该语句下代码块不被运行。
    net = U_Net(1, 2).cuda()
    print(net)
    in1 = torch.randn(1, 1, 64, 64).cuda()
    out = net(in1)
    print(out.size())



