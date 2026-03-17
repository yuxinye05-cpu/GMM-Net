import torch
from torch import nn
from torch.nn import functional as F
from torchstat import stat
import time
import math

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('dropout', nn.Dropout2d(0.1))
    if leaky:
        # stage.add_module('leaky', nn.GELU())
        stage.add_module('leaky', nn.LeakyReLU())
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage

class MSPF(nn.Module):
    def __init__(self, dim_tup=(256, 128, 64,32), rfb=False, vis=False):
        super(MSPF, self).__init__()
        self.dim = dim_tup
        self.dim_0 = self.dim[0]
        self.dim_1 = self.dim[1]
        self.dim_2 = self.dim[2]
        self.dim_3 = self.dim[3]
        self.inter_dim = self.dim[3]

        self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
        self.compress_level_1 = add_conv(self.dim[1], self.inter_dim, 1, 1)
        self.compress_level_2 = add_conv(self.dim[2], self.inter_dim, 1, 1)
        self.expand = add_conv(self.inter_dim, self.dim[3], 3, 1)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2, x_level_3):

        level_0_compressed = self.compress_level_0(x_level_0)
        level_0_resized = F.interpolate(level_0_compressed, scale_factor=8, mode='nearest')
        level_1_compressed = self.compress_level_1(x_level_1)
        level_1_resized = F.interpolate(level_1_compressed, scale_factor=4, mode='nearest')
        level_2_compressed = self.compress_level_2(x_level_2)
        level_2_resized = F.interpolate(level_2_compressed, scale_factor=2, mode='nearest')
        level_3_resized = x_level_3

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)
        # **
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v,level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:3, :, :] + \
                            level_3_resized * levels_weight[:, 3:, :, :]


        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


if __name__ == '__main__':
    x1 = torch.randn(10, 32, 256, 256)
    x2 = torch.randn(10, 64, 128, 128)
    x3 = torch.randn(10, 128, 64, 64)
    x4 = torch.randn(10, 256, 32, 32)
    module = MSPF()
    print(module(x4, x3, x2,x1).shape)
