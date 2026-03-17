import torch
from torch import nn
from torchstat import stat


# 位置注意力模块旨在利用任意两点特征之间的关联，来相互增强各自特征的表达
class PAM_Module(nn.Module):
    """ Position attention module
        1. 特征图A(C×H×W)首先分别通过3个卷积层得到3个特征图B、C、D，然后将B、C、D reshape操作维度变为C×N（N=H×W）。**
        2. 然后将reshape操作后的B经过transpose(NxC)与reshape后的C(CxN)相乘，再通过softmax得到空间注意力S(N×N)**
        3. 接着在reshape后的D(CxN)和S的转置(NxN)之间执行矩阵乘法，再乘以尺度系数α，再reshape为原来形状，
        4. 最后与A相加得到最后的输出E其中α初始化为0，并逐渐的学习得到更大的权重**
    """

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        # x:A  proj_query:B  proj_key:C  proj_value:D attention:S out: E
        # .view() 返回一个有相同数据但大小不同的tensor。
        # .permute() 交换tensor的维度
        # .bmm()对存储在两个批batch1和batch2内的矩阵进行批矩阵乘操作
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module
        第一条线是对A(CxHxW)先做reshape(CxN)再做transpose(NxC)**
        第二条线是对A(CxHxW)做reshape(CxN)，然后和第一条线得到的结果相乘得到的结果做softmax得到得到channel attention map X(C×C)**
        第三线是A(CxHxW)做reshape(CxN)，和第二条线得到的结果相乘的结果做reshape为原来形状**
        第四条线是对A和第三条线得到的结果相加得到E**
    """

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        # .view() 返回一个有相同数据但大小不同的tensor。
        # .permute() 交换tensor的维度
        # .bmm() 对存储在两个批batch1和batch2内的矩阵进行批矩阵乘操作
        # .max() 返回输入张量给定维度上每行的最大值，并同时返回每个最大值的位置索引
        # .expand_as() 将tensor扩展为参数tensor的大小
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class DAAM_Module(nn.Module):

    def __init__(self, in_dim):
        super(DAAM_Module, self).__init__()
        self.sa = PAM_Module(in_dim)
        self.ca = CAM_Module(in_dim)
        self.conv51 = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_dim),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_dim),
                                    nn.ReLU())
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(in_dim),
                                   nn.ReLU())

    def forward(self, x):
        sa_feat = self.sa(x)
        sa_conv = self.conv51(sa_feat)

        sc_feat = self.ca(x)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv + sc_conv
        sasc_output =self.conv8(feat_sum)
        return sasc_output


if __name__ == '__main__':
    img = torch.randn(32, 32, 64, 64)
    model = DAAM_Module(32)
    out = model(img)
    criterion = torch.nn.L1Loss()
    loss = criterion(out, img)
    loss.backward()
    print('out.shape:{}'.format(out.shape))
    print('loss:{}'.format(loss))
    stat(model,(32,64,64))

