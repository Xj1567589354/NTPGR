import torch
import winnt
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from collections import OrderedDict
from torch.nn import functional as F
from lib.models.activation_func import *
from lib.models.epsa import PSAModule
from lib.models.cbam import CBAM
from lib.models.dsconv import DSConv_pro

"""
到时候BasicBlock首先使用SA--CA--PSA，按这种顺序替换看效果，选择效果最佳的即可
"""


# 通道重组
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    # 每组的通道数
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    # 分组
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # [batch_size, channels_per_group, groups, height, width]
    # channel shuffle
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


# 空间注意力
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out = x * out
        return out


# sa注意力模块，对se模块的改进，不仅能够学习通道注意力，还能学习空间注意力，计算量参数量还少
class SqueezeAttention(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(SqueezeAttention, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义了四个可学习的参数，形状为[1, C//2G, 1, 1],Parameter--自动注册为模型的可训练参数
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)        # 分组
        x_0, x_1 = x.chunk(2, dim=1)                    # 针对每组进行二等分

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)                  # 聚合

        out = channel_shuffle(out, 2)
        return out


# SE模块
class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c: int,      # block input channel
                 expand_c: int,     # block expand channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor      # 第一个fc的结点个数
        self.fc1 = nn.Conv2d(in_channels=expand_c, out_channels=squeeze_c, kernel_size=1)
        self.ac1 = nn.SiLU()
        self.fc2 = nn.Conv2d(in_channels=squeeze_c, out_channels=expand_c, kernel_size=1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        out = scale * x
        return out


# 非局部注意力模块
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


# 非局部注意力模块--主要使用这个
class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer,)


# ECA模块
class ECA(nn.Module):
    """Constructs a ECA module.

       Args:
           k: Adaptive selection of kernel size
       """
    def __init__(self, k=5):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=int(k/2), bias=False)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = torch.sigmoid(y)
        # 利用expand_as()方法使得y的形状与x保持一致
        return x*y.expand_as(x)


# MECA模块,基于ECA改进的模块
class MECA(nn.Module):
    """Constructs a ECAM module.
       Args:
           k: Adaptive selection of kernel size
       """
    def __init__(self, k=5):
        super(MECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv0 = nn.Conv1d(1, 1, 1, padding=0)
        self.conv = nn.Conv1d(1, 1, k, padding=int(k/2), bias=False)
        self.bn = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y_avgp = self.avg_pool(x)
        y_maxp = self.max_pool(x)
        y = y_avgp + y_maxp
        y = self.conv0(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.relu(self.bn(y))
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = torch.sigmoid(y)
        # 利用expand_as()方法使得y的形状与x保持一致
        return x*y.expand_as(x)


# 坐标注意力
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
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


# SCA, 改进坐标注意力
class SCoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(SCoordAtt, self).__init__()
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.max_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.max_pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.avg_pool_h(x) + self.max_pool_h(x)
        x_w = (self.avg_pool_w(x) + self.max_pool_w(x)).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


# 极化自注意力
class ParallelPolarizedSelfAttention(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel//2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel//2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel//2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel//2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)                        # bs,c//2,h,w
        channel_wq = self.ch_wq(x)                        # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c//2, -1)        # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)           # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)        # bs,c//2,1,1
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).
                                              permute(0, 2, 1))).permute(0, 2, 1).reshape(b, c, 1, 1)  # bs,c,1,1
        channel_out = channel_weight*x

        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(x)                          # bs,c//2,h,w
        spatial_wq = self.sp_wq(x)                          # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)                   # bs,c//2,1,1
        spatial_wv = spatial_wv.reshape(b, c//2, -1)          # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c//2)    # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)              # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))    # bs,1,h,w
        spatial_out = spatial_weight*x
        out = spatial_out+channel_out
        return out


# SK自注意力
class SKAttention(nn.Module):
    def __init__(self, channel, kernels=[1, 3, 5], reduction=16, group=32, L= 32):
        super().__init__()
        self.d = max(L, channel//reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k//2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        # split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)       # k,bs,channel,h,w

        # fuse
        U = sum(conv_outs)                      # bs,c,h,w

        # reduction channel
        S = U.mean(-1).mean(-1)                 # bs,c
        Z = self.fc(S)                          # bs,d

        # calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))              # bs,channel
        attention_weughts = torch.stack(weights, 0)               # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)       # k,bs,channel,1,1

        # fuse
        V = (attention_weughts*feats).sum(0)
        return V


"""
这里是DANet中提到的位置注意力和通道注意力
如果使用这个需要借鉴DANet当中最后fuse融合方式，详情见原文
"""
class PAM_Module(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
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
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
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
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


# 这里是使用DANet当中的PAM和CAM注意力，实现的Head输出
class PCAtten_Head(nn.Module):
    def __init__(self, in_channels):
        super(PCAtten_Head, self).__init__()
        self.position_attention_module = PAM_Module(in_dim=in_channels)
        self.channel_attention_module = CAM_Module(in_dim=in_channels)

    def forward(self, x):
        p_out = self.position_attention_module(x)
        c_out = self.channel_attention_module(x)
        return p_out+c_out


if __name__ == '__main__':
    input = torch.randn(8, 32, 64, 48)
    # nolocalblock = NoLocalBlock2D(in_channels=input.size(1))
    # block = ParallelPolarizedSelfAttention(channel=32)
    block = CoordAtt(32, 32)
    # block = SCoordAtt(32, 32)
    output = block(input)
    print(output)
