import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Basic Conv Block
# ----------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ----------------------
# Attention Block
# ----------------------
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = in_channels // 2

        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: skip connection, g: decoder feature
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # 对齐 H/W
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.relu(x1 + g1)
        psi = self.psi(psi)

        return x * psi

# ----------------------
# UpBlock
# ----------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 对齐 H/W
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

# ----------------------
# HMAN Model
# ----------------------
class HMAN(nn.Module):
    def __init__(self, in_ch=1, n_classes=2):
        super().__init__()

        # Encoder (简化 UNet 风格)
        self.conv1 = ConvBlock(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.center = ConvBlock(512, 512)

        # Attention Blocks
        self.att4 = AttentionBlock(in_channels=512, gating_channels=512)
        self.att3 = AttentionBlock(in_channels=256, gating_channels=512)
        self.att2 = AttentionBlock(in_channels=128, gating_channels=256)
        self.att1 = AttentionBlock(in_channels=64, gating_channels=128)

        # Decoder
        self.up1 = UpBlock(in_ch=512, skip_ch=512, out_ch=512)
        self.up2 = UpBlock(in_ch=512, skip_ch=256, out_ch=256)
        self.up3 = UpBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.up4 = UpBlock(in_ch=128, skip_ch=64, out_ch=64)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))
        x5 = self.center(self.pool4(x4))

        # Attention
        x4 = self.att4(x4, x5)
        x3 = self.att3(x3, x5)
        x2 = self.att2(x2, x3)
        x1 = self.att1(x1, x2)

        # Decoder
        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)

        out = self.final(d1)
        return out
