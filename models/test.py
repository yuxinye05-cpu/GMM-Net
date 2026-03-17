import torch
from torch import nn
from torch.nn import functional as F
from torchstat import stat
from .MSI import MultiScaleInputModule
from .cloformer import EfficientBlock
from .MCAM import MCAM
from .MSPF import MSPF

import time


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel,drop_rate=0.25,attention=False,):
        super(Conv_Block, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1,stride=1)
        self.layer = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
            nn.Dropout2d(drop_rate),
        )
        if attention:
            self.layer_1=nn.Sequential(
                                       nn.LeakyReLU()
                                       )
        else:
            self.layer_1=nn.Identity()
        self.GELU = nn.GELU()

    def forward(self, x):
        x = self.conv1x1(x)
        out = self.layer(x)
        # out = self.GELU(out + x)
        # out = self.layer_1(out)
        return out

class Conv_Block_1(nn.Module):
    def __init__(self, in_channel, out_channel,attention=False):
        super(Conv_Block_1, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1,stride=1)
        self.layer = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.25),
            nn.GELU(),

        )

        #self.eca = ECABlock(out_channel)
        self.GELU = nn.GELU()

    def forward(self, x):
        x = self.conv1x1(x)
        #out = self.layer(x)
        out = self.GELU(x)
        #out = self.layer_1(out)
        return out


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化
            nn.BatchNorm2d(channel),
            nn.GELU()
        )

    def forward(self, x):
        return self.layer(x)


class Conv1x1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv1x1, self).__init__()
        self.layer = nn.Conv2d(in_channel, out_channel, kernel_size=1,stride=1)

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(in_channel, out_channel, 1, 1)

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)  # 转置卷积会出现空洞，影响图像分割，所以采用插值法
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


class Test(nn.Module):
    def __init__(self, num_classes=2):
        super(Test, self).__init__()
        r = 2
        self.c1 = Conv_Block(1, 16 * r,0.1)
        self.d1 = DownSample(16 * r)
        self.c2 = Conv_Block(32 * r, 32 * r,0.2)
        self.d2 = DownSample(32 * r)
        self.c3 = Conv_Block(64 * r, 64 * r)
        self.d3 = DownSample(64 * r)
        self.c4 = Conv_Block_1(128 * r, 128 * r)

        self.cv_1 = Conv1x1(64 * r, 128 * r)

        self.att1 = MCAM(16 * r)
        self.att2 = MCAM(32 * r)
        self.att3 = MCAM(64 * r)

        self.former= EfficientBlock(128 * r, 128 * r, 8, [4, 4], [3], 2, 3, 4)
        self.former_1 = EfficientBlock(128 * r, 128 * r, 8, [4, 4], [3], 2, 3, 4)

        self.msi1 = MultiScaleInputModule(16 * r,1)
        self.msi2 = MultiScaleInputModule(32 * r, 2)
        self.msi3 = MultiScaleInputModule(64 * r, 3)

        self.u1 = UpSample(128 * r, 64 * r)
        self.c6 = Conv_Block(128 * r, 64 * r)
        self.u2 = UpSample(64 * r, 32 * r)
        self.c7 = Conv_Block(64 * r, 32 * r)
        self.u3 = UpSample(32 * r, 16 * r)
        self.c8 = Conv_Block(32 * r, 16 * r,0.2)
        self.u4 = UpSample(32 * r, 16 * r)
        self.c9 = Conv_Block(32 * r, 16 * r)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.mspf=MSPF((128*r,64* r,32* r,16* r))
        self.out = nn.Conv2d(16 * r, num_classes, 1)
        self.out_layer = nn.Sequential(
            nn.Conv2d(16 * r, num_classes, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x1=x
        R1 = self.c1(x)
        R2 = self.c2(self.msi1(x1,self.d1(R1)))
        R3 = self.c3(self.msi2(x1,self.d2(R2)))
        #R3 = self.attention1(R3)
        R4 = self.c4(self.msi3(x1,self.d3(R3)))

        R4 = self.former(R4)

        F0 = self.att3(R3)
        F1 = self.att2(R2)
        F2 = self.att1(R1)

        #D1 = self.attention3(self.c6(self.u1(R4, F0)))
        D1 = self.c6(self.u1(R4, F0))
        D2 = self.c7(self.u2(D1, F1))
        D3 = self.c8(self.u3(D2, F2))

        # F3 = self.ASFF3(R4, D1, D2)
        # D4 = self.c9(self.u4(F3, D3))
        # D4= D3+D4
        D3=self.mspf(R4,D1,D2,D3)+D3

        return F.softmax(self.out(D3), dim=1)



if __name__ == '__main__':
    x = torch.randn(1, 1, 128, 128)
    net = Test()
    print(net(x).shape)
    stat(net,(1,128,128))
    time1=time.time()
    for i in range(10):
      out = net(x)
    time2=time.time()
    print(out.size())
    stat(net, (1, 128, 128))
    print('Inference time:',(time2 - time1)/10)