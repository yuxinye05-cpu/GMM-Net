import torch
from torch import nn
from torch.nn import functional as F


# 实现Sobel梯度计算
class SobelGradient(nn.Module):
    def __init__(self):
        super(SobelGradient, self).__init__()
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        # 计算梯度
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        # 计算梯度幅值
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        # 归一化
        grad_magnitude = (grad_magnitude - grad_magnitude.min()) / (grad_magnitude.max() - grad_magnitude.min() + 1e-6)
        return grad_magnitude



class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.expand = expand

        # 极简结构
        self.conv1 = nn.Conv2d(dim, dim * expand, 3, 1, 1, groups=dim)
        self.conv2 = nn.Conv2d(dim * expand, dim, 1, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.activation = nn.GELU()

        # 简单的门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        # 门控
        gate = self.gate(x)

        # 深度可分离卷积
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)

        # 应用门控
        x = x * gate

        # 残差连接
        x = self.norm(x + identity)

        return x


# 多尺度状态模块 (MSM)
class MultiScaleStateModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MultiScaleStateModule, self).__init__()

        # 局部卷积分支
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
        )

        # 全局状态空间分支 (Mamba)
        self.mamba_branch = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            MambaBlock(out_channel),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
        )

        # 梯度调制
        self.conv_weight = nn.Conv2d(1, in_channel, 1, 1)
        self.mamba_weight = nn.Conv2d(1, in_channel, 1, 1)

        # 融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, gradient):
        # 调整梯度图大小以匹配特征图
        if gradient.shape[2:] != x.shape[2:]:
            gradient = F.interpolate(gradient, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 计算梯度权重
        w_conv = torch.sigmoid(self.conv_weight(gradient))
        w_mamba = torch.sigmoid(self.mamba_weight(gradient))

        # 分支处理
        f_conv = self.conv_branch(x * w_conv)
        f_mamba = self.mamba_branch(x * w_mamba)

        # 自适应融合
        lambda_weight = torch.sigmoid(self.fusion_weight)
        return lambda_weight * f_conv + (1 - lambda_weight) * f_mamba


# 梯度引导轻量Transformer (GLT)
class GradientGuidedLightweightTransformer(nn.Module):
    def __init__(self, dim):
        super(GradientGuidedLightweightTransformer, self).__init__()
        self.dim = dim

        # 简单的卷积处理
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # 通道注意力（简化）
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        # 梯度调制
        self.grad_attn = nn.Conv2d(1, dim, 1, 1)

    def forward(self, x, gradient):
        identity = x

        # 调整梯度图大小以匹配特征图
        if gradient.shape[2:] != x.shape[2:]:
            gradient = F.interpolate(gradient, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 卷积处理
        x = self.conv(x)

        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca

        # 梯度调制
        grad_weight = torch.sigmoid(self.grad_attn(gradient))
        x = x * grad_weight

        # 残差连接
        return x + identity


# 门控跳跃融合模块 (GSF)
class GatedSkipFusion(nn.Module):
    def __init__(self, encoder_ch, decoder_ch):
        super(GatedSkipFusion, self).__init__()
        self.encoder_proj = nn.Conv2d(encoder_ch, decoder_ch, 1)

        # 简单门控
        self.gate = nn.Sequential(
            nn.Conv2d(decoder_ch, decoder_ch, 1),
            nn.Sigmoid()
        )

        self.fusion = nn.Conv2d(decoder_ch * 2, decoder_ch, 3, 1, 1)

    def forward(self, encoder_feat, decoder_feat):
        encoder_proj = self.encoder_proj(encoder_feat)
        gate = self.gate(decoder_feat)
        gated_encoder = encoder_proj * gate
        fused = torch.cat([gated_encoder, decoder_feat], dim=1)
        return self.fusion(fused)


# 修复版多尺度特征桥 (MFB)
class MultiScaleFeatureBridge(nn.Module):
    def __init__(self, channels_list, out_channel):
        super(MultiScaleFeatureBridge, self).__init__()

        # 计算每个特征需要的上采样倍数
        # 假设输入特征列表按从小（深层）到大（浅层）排列
        # d3 (16*r, H, W) - 最大，不需要上采样
        # d2 (32*r, H/2, W/2) - 上采样2倍
        # d1 (64*r, H/4, W/4) - 上采样4倍
        # b (128*r, H/8, W/8) - 上采样8倍

        self.num_scales = len(channels_list)

        # 创建上采样模块，每个特征上采样到原始输入大小
        self.up_convs = nn.ModuleList()
        for i, ch in enumerate(channels_list):
            # 第i个特征需要上采样 2^i 倍
            scale_factor = 2 ** i
            self.up_convs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                    nn.Conv2d(ch, out_channel, 1)
                )
            )

        # 融合卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channel * self.num_scales, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
        )

    def forward(self, features_list, gradient):
        # 假设features_list按从浅层到深层排列：[d3, d2, d1, b]
        # d3: 原始大小 (16*r, H, W)
        # d2: 下采样一次 (32*r, H/2, W/2)
        # d1: 下采样两次 (64*r, H/4, W/4)
        # b: 下采样三次 (128*r, H/8, W/8)

        upsampled = []
        for i, (feat, up_conv) in enumerate(zip(features_list, self.up_convs)):
            upsampled_feat = up_conv(feat)
            upsampled.append(upsampled_feat)

        # 拼接所有特征
        fused = torch.cat(upsampled, dim=1)

        # 融合卷积
        output = self.fusion(fused)

        return output


# 下采样模块
class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1),  # 使用stride=2的下采样
            nn.BatchNorm2d(channel),
            nn.GELU()
        )

    def forward(self, x):
        return self.down(x)


# 上采样模块
class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.up(x)


# GMM-Net 主网络
class GMMNet(nn.Module):
    def __init__(self, num_classes=2, r=2):
        super(GMMNet, self).__init__()

        # Sobel梯度计算
        self.sobel = SobelGradient()

        # 编码器路径
        self.encoder1 = MultiScaleStateModule(1, 16 * r)
        self.down1 = DownSample(16 * r)

        self.encoder2 = MultiScaleStateModule(16 * r, 32 * r)
        self.down2 = DownSample(32 * r)

        self.encoder3 = MultiScaleStateModule(32 * r, 64 * r)
        self.down3 = DownSample(64 * r)

        self.encoder4 = MultiScaleStateModule(64 * r, 128 * r)

        # 瓶颈 - 梯度引导轻量Transformer
        self.glt = GradientGuidedLightweightTransformer(128 * r)

        # 解码器路径
        self.up1 = UpSample(128 * r, 64 * r)
        self.gsf1 = GatedSkipFusion(64 * r, 64 * r)
        self.decoder1 = MultiScaleStateModule(64 * r, 64 * r)

        self.up2 = UpSample(64 * r, 32 * r)
        self.gsf2 = GatedSkipFusion(32 * r, 32 * r)
        self.decoder2 = MultiScaleStateModule(32 * r, 32 * r)

        self.up3 = UpSample(32 * r, 16 * r)
        self.gsf3 = GatedSkipFusion(16 * r, 16 * r)
        self.decoder3 = MultiScaleStateModule(16 * r, 16 * r)

        # 多尺度特征桥
        self.mfb = MultiScaleFeatureBridge(
            channels_list=[16 * r, 32 * r, 64 * r, 128 * r],
            out_channel=16 * r
        )

        # 输出层
        self.out_conv = nn.Conv2d(16 * r, num_classes, 1)

    def forward(self, x):
        # 计算梯度先验
        gradient = self.sobel(x)

        # 编码器路径
        e1 = self.encoder1(x, gradient)  # (B, 16*r, H, W)
        e2 = self.encoder2(self.down1(e1), gradient)  # (B, 32*r, H/2, W/2)
        e3 = self.encoder3(self.down2(e2), gradient)  # (B, 64*r, H/4, W/4)
        e4 = self.encoder4(self.down3(e3), gradient)  # (B, 128*r, H/8, W/8)

        # 瓶颈处理
        b = self.glt(e4, gradient)  # (B, 128*r, H/8, W/8)

        # 解码器路径
        d1_up = self.up1(b)  # (B, 64*r, H/4, W/4)
        d1_cat = self.gsf1(e3, d1_up)  # (B, 64*r, H/4, W/4)
        d1 = self.decoder1(d1_cat, gradient)  # (B, 64*r, H/4, W/4)

        d2_up = self.up2(d1)  # (B, 32*r, H/2, W/2)
        d2_cat = self.gsf2(e2, d2_up)  # (B, 32*r, H/2, W/2)
        d2 = self.decoder2(d2_cat, gradient)  # (B, 32*r, H/2, W/2)

        d3_up = self.up3(d2)  # (B, 16*r, H, W)
        d3_cat = self.gsf3(e1, d3_up)  # (B, 16*r, H, W)
        d3 = self.decoder3(d3_cat, gradient)  # (B, 16*r, H, W)

        # 多尺度特征桥
        # 注意：这里假设d3, d2, d1, b的尺寸分别是：H, H/2, H/4, H/8
        features_list = [d3, d2, d1, b]
        final_feat = self.mfb(features_list, gradient)

        # 输出
        out = self.out_conv(final_feat)

        return out


# 测试函数
def test_model():
    print("Testing GMMNet...")

    # 创建模型
    model = GMMNet(r=2)

    # 测试不同输入尺寸
    test_sizes = [(16, 1, 128, 128), (4, 1, 160, 160), (1, 1, 256, 256)]

    for size in test_sizes:
        print(f"\nTesting with input size: {size}")
        x = torch.randn(size)

        try:
            output = model(x)
            print(f"✓ Success! Output shape: {output.shape}")

            # 打印中间特征尺寸
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")

        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


if __name__ == '__main__':
    model = test_model()