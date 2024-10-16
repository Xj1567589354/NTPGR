import torch
import torch.nn as nn
import math

from lib.models.cbam import *
from lib.models.condconv import *
from lib.models.block import *

BN_MOMENTUM = 0.1
num_channels = [32, 64, 128, 256]


class convT(nn.Module):
    def __init__(self, in_planes, planes, k=4, s=2, p=1, b=False):
        super(convT, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels=in_planes, out_channels=planes, kernel_size=k,
                                         stride=s, padding=p, bias=b)
        self.bn = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_t(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


# 混合高效通道注意力
class MECA(nn.Module):
    def __init__(self, channels=64, r=4, k=5,
                 pool_types=['avg', 'max']):
        super(MECA, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = Optim_ChannelGate(k=k, pool_types=pool_types)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


# ECA, 改进坐标注意力
class ECoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(ECoordAtt, self).__init__()
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.max_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.max_pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        # self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.conv1 = CondConv2D(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        # self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.conv_h = CondConv2D(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = CondConv2D(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()

        x_avg_h = self.avg_pool_h(x)
        x_avg_w = self.avg_pool_w(x).permute(0, 1, 3, 2)
        x_avg = torch.cat([x_avg_h, x_avg_w], dim=2)

        x_max_h = self.max_pool_h(x)
        x_max_w = self.max_pool_w(x).permute(0, 1, 3, 2)
        x_max = torch.cat([x_max_h, x_max_w], dim=2)

        y = x_avg + x_max
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


# 注意力特征融合模块
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()

        self.mscam = MS_CAM(channels=channels, r=r)

    def forward(self, x, residual):
        xa = x + residual
        wei = self.mscam(xa)

        xo = x * wei + residual * (1 - wei)
        return xo


# 新型注意力特征融合模块
class NAFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(NAFF, self).__init__()

        self.mscam = MECA(channels=channels, r=r)

    def forward(self, x, residual):
        xa = x + residual
        wei = self.mscam(xa)

        xo = x * wei + residual * (1 - wei)
        return xo


# 第一条分支是目标分支
class PFFM(nn.Module):
    def __init__(self):
        super(PFFM, self).__init__()

        self.convT1 = convT(in_planes=num_channels[3], planes=num_channels[2])
        self.convT2 = convT(in_planes=num_channels[2], planes=num_channels[1])
        self.convT3 = convT(in_planes=num_channels[1], planes=num_channels[0])

        self.naff1 = NAFF(channels=num_channels[2])
        self.naff2 = NAFF(channels=num_channels[1])
        self.naff3 = NAFF(channels=num_channels[0])

        # 再次加强特征融合
        self.nf = nn.Sequential(
            ConvBNActivation(in_planes=num_channels[0], out_planes=num_channels[0], kernel_size=3,
                             stride=1, groups=num_channels[0]),
            GhostModule(inp=num_channels[0], oup=num_channels[0], kernel_size=1, relu=True,
                        relu_forward=False, silu_forward=True),
            ECoordAtt(inp=num_channels[0], oup=num_channels[0])
        )

    def forward(self, x1, x2, x3, x4):
        out = self.convT1(x1)
        out = self.naff1(out, x2)

        out = self.convT2(out)
        out = self.naff2(out, x3)

        out = self.convT3(out)
        out = self.naff3(out, x4)

        out = self.nf(out)

        return out


# 第二条分支是目标分支
class PFFM_2(nn.Module):
    def __init__(self, hsize, wsize):
        super(PFFM_2, self).__init__()

        self.convT1 = convT(in_planes=num_channels[3], planes=num_channels[2])
        self.convT2 = convT(in_planes=num_channels[2], planes=num_channels[1])

        self.avgpool = nn.AdaptiveAvgPool2d((hsize // 2, wsize // 2))
        self.conv3 = GhostModule(inp=num_channels[0], oup=num_channels[1], kernel_size=1, relu=True,
                                 relu_forward=False, silu_forward=True)

        self.naff1 = NAFF(channels=num_channels[2])
        self.naff2 = NAFF(channels=num_channels[1])

        # 再次加强特征融合
        self.nf = nn.Sequential(
            ConvBNActivation(in_planes=num_channels[0], out_planes=num_channels[0], kernel_size=3,
                             stride=1, groups=num_channels[0]),
            GhostModule(inp=num_channels[0], oup=num_channels[0], kernel_size=1, relu=True,
                        relu_forward=False, silu_forward=True),
            ECoordAtt(inp=num_channels[0], oup=num_channels[0])
        )

    def forward(self, x1, x2, x3, x4):
        out = self.convT1(x1)
        out = self.naff1(out, x2)

        out = self.convT2(out)
        out1 = self.naff2(out, x3)

        out2 = self.conv3(self.avgpool(x4))
        out = self.naff2(out1, out2)

        out = self.nf(out)

        return out

if __name__ == '__main__':
    input = torch.randn(8, 32, 64, 48)
    # block = MECA(channels=32)
    # block = AFF(channels=32)
    # block = NAFF(channels=32)
    block = ECoordAtt(32, 32)
    output = block(input)
    print(output)