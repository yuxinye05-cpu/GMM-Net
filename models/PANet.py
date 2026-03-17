import torch
from torch import nn
from torch.nn import functional as F
from torchstat import stat
import time
import math

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel,attention=False):
        super(Conv_Block, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1,stride=1)
        self.layer = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
            nn.Dropout2d(0.25),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
            nn.Dropout2d(0.25),
        )
        if attention:
            self.layer_1=nn.Sequential(
                                       nn.LeakyReLU()
                                       )
        else:
            self.layer_1=nn.Identity()
        #self.eca = ECABlock(out_channel)
        self.GELU = nn.GELU()

    def forward(self, x):
        x = self.conv1x1(x)
        out = self.layer(x)
        out = self.GELU(out + x)
        out = self.layer_1(out)
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
        self.c1 = Conv_Block(1, 16 * r)
        self.d1 = DownSample(16 * r)
        self.c2 = Conv_Block(16 * r, 32 * r)
        self.d2 = DownSample(32 * r)
        self.c3 = Conv_Block(32 * r, 64 * r)
        self.d3 = DownSample(64 * r)
        self.c4 = Conv_Block_1(64 * r, 128 * r)

        self.cv_1 = Conv1x1(64 * r, 128 * r)

        #self.attention1 = DAAM_up_2(64 * r, 8)
        self.attention2 = DAAM_up(128 * r, 8)
        #self.attention3 = DAAM_up_2(64 * r, 8)

        self.ASFF0 = ASFF(0, (64 * r, 32 * r, 16 * r))
        self.ASFF1 = ASFF(1, (64 * r, 32 * r, 16 * r))
        self.ASFF2 = ASFF(2, (64 * r, 32 * r, 16 * r))
        self.ASFF3 = ASFF(2, (128 * r, 64 * r, 32 * r))

        self.u1 = UpSample(128 * r, 64 * r)
        self.c6 = Conv_Block(128 * r, 64 * r)
        self.u2 = UpSample(64 * r, 32 * r)
        self.c7 = Conv_Block(64 * r, 32 * r)
        self.u3 = UpSample(32 * r, 16 * r)
        self.c8 = Conv_Block(32 * r, 16 * r)
        self.u4 = UpSample(32 * r, 16 * r)
        self.c9 = Conv_Block(32 * r, 16 * r)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.out = nn.Conv2d(16 * r, num_classes, 1)

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        #R3 = self.attention1(R3)
        R4 = self.c4(self.d3(R3))

        R4 = self.attention2(R4)

        F0 = self.ASFF0(R3, R2, R1)
        F1 = self.ASFF1(R3, R2, R1)
        F2 = self.ASFF2(R3, R2, R1)

        #D1 = self.attention3(self.c6(self.u1(R4, F0)))
        D1 = self.c6(self.u1(R4, F0))
        D2 = self.c7(self.u2(D1, F1))
        D3 = self.c8(self.u3(D2, F2))

        F3 = self.ASFF3(R4, D1, D2)
        D4 = self.c9(self.u4(F3, D3))
        D4= D3+D4

        return F.softmax(self.out(D4), dim=1)

class AxialAttention_dynamic(nn.Module):
    def __init__(self, in_dim, groups=8, kernel_size=16,
                 stride=1, bias=False, width=False):
        assert (in_dim % groups == 0) and (in_dim % groups == 0)
        super(AxialAttention_dynamic, self).__init__()
        self.in_planes = in_dim
        self.out_planes = in_dim
        self.groups = groups
        self.group_planes = in_dim // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = nn.Conv1d(in_dim, in_dim * 2, kernel_size=1, stride=1,
                                       padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(in_dim * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(in_dim * 2)

        # Priority on encoding

        ## Initial values

        self.f_qr = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))  # 模型的常数，不参与反向传播，生成索引
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()
        # self.print_para()

    def forward(self, x):
        #决定使用横轴自注意力还是纵轴自注意力
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        # 通过数组索引的方式生成相对位置编码
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        all_embeddings=F.interpolate(all_embeddings.unsqueeze(0),size=(H,H),mode='bicubic',align_corners=True).squeeze(0)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)
        # 可删除
        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class CAM_Module(nn.Module):
    """ Channel attention module
        第一条线是对A(CxHxW)先做reshape(CxN)再做transpose(NxC)**
        第二条线是对A(CxHxW)做reshape(CxN)，然后和第一条线得到的结果相乘得到的结果做softmax得到得到channel attention map X(C×C)**
        第三线是A(CxHxW)做reshape(CxN)，和第二条线得到的结果相乘的结果做reshape为原来形状**
        第四条线是对A和第三条线得到的结果相加得到E**
    """

    def __init__(self, in_dim,groups=8,kernel_size=4):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.groups=groups
        self.group_planes=kernel_size*kernel_size//groups
        self.qkv_transform = nn.Conv1d(in_dim, in_dim* 3, kernel_size=1, stride=1,
                                       padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(kernel_size* kernel_size)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(kernel_size* kernel_size*2)


        self.relative = nn.Parameter(torch.randn(3,self.chanel_in * 2 - 1), requires_grad=True)
        query_index = torch.arange(self.chanel_in).unsqueeze(0)
        key_index = torch.arange(self.chanel_in).unsqueeze(1)
        relative_index = key_index - query_index + self.chanel_in - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        self.f_qr = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.f_kr = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.f_sve = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.f_sv = nn.Parameter(torch.tensor(1.0), requires_grad=True)


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):

        # .view() 返回一个有相同数据但大小不同的tensor。
        # .permute() 交换tensor的维度
        # .bmm() 对存储在两个批batch1和batch2内的矩阵进行批矩阵乘操作
        # .max() 返回输入张量给定维度上每行的最大值，并同时返回每个最大值的位置索引
        # .expand_as() 将tensor扩展为参数tensor的大小
        N, C, H, W = x.size()
        x1=x.view(N,C,H*W)
        group_planes=H*W//self.groups
        x1=self.qkv_transform(x1)
        x1=F.layer_norm(x1,[H*W])  #发现代码写错，回来补救

        q, k, v = torch.split(x1.contiguous().reshape(N, self.groups, C*3, -1),[C, C, C], dim=2)

        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(3,C,C)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,[1, 1,1], dim=0)
        q_embedding=q_embedding.repeat(group_planes,1,1)
        k_embedding=k_embedding.repeat(group_planes,1,1)
        v_embedding=v_embedding.repeat(group_planes,1,1)

        qr = torch.einsum('bgci,icj->bgcj', q, q_embedding)
        kr = torch.einsum('bgci,icj->bgcj', k, k_embedding).transpose(2, 3)
        qk=torch.einsum('bgci,bgji->bgcj',q,k)

        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)


        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N , 3, self.groups, C, C).sum(dim=1)
        # 此处可再修改一下
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgci,bgij->bgjc', similarity, v)
        sve = torch.einsum('bgci,jci->bgjc', similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N,H*W*2, C)
        output = F.layer_norm(stacked_output.permute(0,2,1),[H*W*2]).permute(0,2,1).contiguous()
        output=output.view(N, self.groups, group_planes, 2, C).sum(dim=-2)

        out = output.view(N, C, H, W)

        out = self.gamma * out + x
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
            nn.Dropout2d(drop)
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
            nn.Dropout2d(drop)
        )
        self.drop = nn.Dropout2d(drop)
        self.bn=nn.Sequential(
            nn.GELU(),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        #B, N, C = x.shape
        #x = x.permute(0, 2, 1).reshape(B, C, H, W)
        shortcut=x
        x = self.conv1(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        return self.bn(x+shortcut)



class DAAM_up(nn.Module):

    def __init__(self, in_dim,groups=8,width=True):
        super(DAAM_up, self).__init__()
        self.aah = AxialAttention_dynamic(in_dim=in_dim,groups=groups)
        self.aaw=AxialAttention_dynamic(in_dim=in_dim,groups=groups,width=width)
        self.ca = CAM_Module(in_dim=in_dim,groups=groups)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.dwconv_0 = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=False,groups=in_dim),
                                    nn.BatchNorm2d(in_dim),
                                    nn.GELU(),
                                    nn.Dropout2d(0.1, False))
        self.bn = nn.Sequential(nn.BatchNorm2d(in_dim),
                                nn.GELU())
        self.bn_1 = nn.Sequential(nn.BatchNorm2d(in_dim),
                                nn.GELU())
        self.dwconv = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=False,groups=in_dim),
                                    nn.BatchNorm2d(in_dim),
                                    nn.GELU(),
                                    nn.Dropout2d(0.1, False))
        self.conv52 = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_dim),
                                    nn.ReLU())
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(in_dim),
                                   nn.ReLU())
        self.layer = nn.Sequential(
                                   nn.GELU(),
                                   nn.BatchNorm2d(in_dim),
                                   nn.Dropout2d(0.1, False),)

        self.mlp=Mlp(in_features=in_dim,hidden_features=4*in_dim)

    def forward(self, x):
        x_1=self.dwconv_0(x)
        x=x_1+x
        x=self.bn(x)
        aa_feat = self.aah(x)
        aa_feat=self.aaw(aa_feat)
        aa_feat=aa_feat+self.gamma*self.dwconv(x)
        aa_feat=self.bn_1(aa_feat)
        aa_feat=aa_feat+x
        aa_feat = self.layer(aa_feat)
        aa_feat=self.mlp(aa_feat)
        #aa_conv = self.conv51(aa_feat)

        #sc_feat = self.ca(x)
        #sc_conv = self.conv52(sc_feat)

        #feat_sum = aa_conv + sc_conv
        #feat_sum=aa_feat+sc_feat
        #sasc_output =self.conv8(feat_sum)
        return aa_feat

class DAAM_up_2(nn.Module):

    def __init__(self, in_dim,groups=8,width=True):
        super(DAAM_up_2, self).__init__()
        self.DAAM_up_0=DAAM_up(in_dim,groups,width)
        self.DAAM_up_1 = DAAM_up(in_dim, groups, width)

    def forward(self, x):

        return self.DAAM_up_1(self.DAAM_up_0(x))

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
    if leaky:
        stage.add_module('leaky', nn.GELU())
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class ASFF(nn.Module):
    def __init__(self, level, dim_tup=(128, 64, 32), rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = dim_tup
        self.dim_0 = self.dim[0]
        self.dim_1 = self.dim[1]
        self.dim_2 = self.dim[2]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(self.dim[1], self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(self.dim[2], self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, self.dim[0], 3, 1)
        elif level == 1:
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(self.dim[2], self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, self.dim[1], 3, 1)
        elif level == 2:
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
            self.compress_level_1 = add_conv(self.dim[1], self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, self.dim[2], 3, 1)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


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