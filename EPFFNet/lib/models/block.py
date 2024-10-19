
from typing import Optional, Callable

import torch

from lib.models.attention_modules import *
from lib.models.ghost import *
from lib.models.epsa import PSAModule


BN_MOMENTUM = 0.2


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# DW卷积或者普通卷积
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 dilation_rate: int = 1,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU

        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False,
                                                         dilation=dilation_rate),
                                               norm_layer(out_planes),
                                               activation_layer(inplace=True))


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )


"""
以下两个模块BasicBlock和Bottleneck模块都是在HRNet原始基本块的基础上融入了sa注意力
"""
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

        if inplanes > 32:
            self.sa = SqueezeAttention(channel=planes, groups=32)
        else:
            self.sa = SqueezeAttention(channel=planes, groups=16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 在两个3*3conv之后加入sa模块
        out = self.sa(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # self.sa = SqueezeAttention(planes, groups=32)
        self.epsa = PSAModule(inplans=planes, planes=planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.epsa(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


"""
改进版BasicBlock，放在反卷积之后，融入了MBConv3block
只是将MBConv3block当中的se替换成了sa模块
如果dilation_rate=2就是将dw卷积融入了空洞卷积的思想，增大感受野，实现空洞dw卷积。否则还是dw卷积
这样会导致block计算量增加，具体还得看实验效果如何
在这里融合了空洞卷积思想，所有stride要等于1
"""
class MBConvBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation_rate=1):
        super(MBConvBasicBlock, self).__init__()
        self.branch1 = nn.Sequential(
            # expand conv
            ConvBNActivation(in_planes=inplanes,
                             out_planes=planes*4,
                             kernel_size=1),
            # dw conv
            ConvBNActivation(in_planes=planes*4,
                             out_planes=planes*4,
                             kernel_size=3,
                             stride=stride,
                             groups=planes*4,
                             dilation_rate=dilation_rate),
            # sa
            SqueezeAttention(channel=planes*4),
            # linear pw conv
            ConvBNActivation(in_planes=planes*4,
                             out_planes=inplanes,
                             kernel_size=1,
                             activation_layer=nn.Identity)
        )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        # x1, x2 = x.chunk(2, dim=1)

        out = self.branch1(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out = torch.cat((out1, out2), dim=1)
        # # group设置为2
        # out = channel_shuffle(out, 2)

        out += residual

        return out


class MBConvDilaBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, kernel_size=3,
                 downsample=None, dilation_rate=1):
        super(MBConvDilaBasicBlock, self).__init__()
        self.branch1 = nn.Sequential(
            # expand conv
            ConvBNActivation(in_planes=inplanes // 2,
                             out_planes=planes*4,
                             kernel_size=1),
            # dila dw conv
            ConvBNActivation(in_planes=planes*4,
                             out_planes=planes*4,
                             kernel_size=kernel_size,
                             stride=stride,
                             groups=planes*4
                             ),
            # # se
            # SqueezeExcitation(input_c=inplanes,
            #                   expand_c=planes*4),
            # sa
            SqueezeAttention(channel=planes * 4),
            # linear pw conv
            ConvBNActivation(in_planes=planes*4,
                             out_planes=inplanes // 2,
                             kernel_size=1,
                             activation_layer=nn.Identity)
        )
        self.branch2 = nn.Identity()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)

        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        out = torch.cat((out1, out2), dim=1)

        # group设置为2
        out = channel_shuffle(out, 2)

        return out


# 改进版BasicBlock
class FusedBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FusedBlock, self).__init__()
        self.branch1 = nn.Sequential(
            # pw
            GhostModule(inp=inplanes//2, oup=planes*4,
                        kernel_size=1, relu=True,
                        relu_forward=False, silu_forward=True),
            # dw conv
            ConvBNActivation(in_planes=planes*4,
                             out_planes=planes*4,
                             kernel_size=3,
                             stride=stride,
                             groups=planes*4),
            # sa
            SqueezeAttention(channel=planes*4),
            # CoordAtt(inp=planes*4, oup=planes*4),
            # ParallelPolarizedSelfAttention(channel=planes*4),
            # SKAttention(channel=planes*4, reduction=8),
            # linear pw conv
            GhostModule(inp=planes*4, oup=inplanes//2, kernel_size=1,
                        relu=False, relu_forward=False, silu_forward=False)
        )
        self.branch2 = nn.Identity()
        # self.downsample = downsample

    def forward(self, x):
        # out1 = self.branch1(x)
        # if self.downsample is not None:
        #     out2 = self.downsample(x)
        # else:
        #     out2 = self.branch2(x)
        # out = out1 + out2

        x1, x2 = x.chunk(2, dim=1)

        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        out = torch.cat((out1, out2), dim=1)

        # group设置为2
        out = channel_shuffle(out, 2)

        return out

"""
以下两个模块GhostBasicBlock和GhostBottleBlock模块是在HRNet基本块的基础上使用ghost卷积替换1*1卷积
并且在GhostBasicBlock是沿用了BasicBlock基本结构，加入了CA注意力
而GhostBottleBlock则是借鉴ghost module，并加入了EPSA模块
"""
# ghost改进BasicBlock
class GhostBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(GhostBasicBlock, self).__init__()
        self.conv1 = GhostModule(inp=inplanes, oup=planes, stride=stride,
                                 kernel_size=3, relu=True)
        self.conv2 = GhostModule(inp=planes, oup=inplanes, stride=stride,
                                 kernel_size=3, relu=False)
        if inplanes > 32:
            self.sa = SqueezeAttention(channel=inplanes, groups=32)
        else:
            self.sa = SqueezeAttention(channel=inplanes, groups=16)
        # self.sk = SKAttention(channel=32, reduction=8)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        # self.ca = CoordAtt(inp=inplanes, oup=inplanes)

    def forward(self, x):
        residual = x
        # x1, x2 = x.chunk(2, dim=1)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.sa(out)
        # out = self.ca(out1)
        # out = self.sk(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        # out = torch.cat((out1, out2), dim=1)

        out = self.relu(out)

        # # group设置为2
        # out = channel_shuffle(out, 2)

        return out


# 改进版Bottleneck
class GhostBottleBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(GhostBottleBlock, self).__init__()
        self.branch1 = nn.Sequential(
            # # expand conv
            GhostModule(inp=inplanes, oup=planes,
                        kernel_size=1, relu=True),
            # dw conv
            # ConvBNActivation(in_planes=planes,
            #                  out_planes=planes,
            #                  kernel_size=3,
            #                  stride=stride,
            #                  groups=planes),
            depthwise_conv(inp=planes, oup=planes, kernel_size=3,
                           stride=stride, relu=False),
            # # se
            # SqueezeExcitation(input_c=inplanes,
            #                   expand_c=planes),
            # # sa
            # SqueezeAttention(channel=planes, groups=32),
            PSAModule(inplans=planes, planes=planes),
            # # linear pw conv
            GhostModule(inp=planes, oup=planes*4, kernel_size=1, relu=False)
        )
        self.branch2 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out1 = self.branch1(x)

        if self.downsample is not None:
            out2 = self.downsample(x)
        else:
            out2 = x

        out = out1 + out2

        return out


# 结合IBN-Net的思想改进的Bottleneck
class GLBottleBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(GLBottleBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # 批量归一化和实例归一化
        self.bn1 = nn.BatchNorm2d(planes//2, momentum=BN_MOMENTUM)
        self.ln1 = nn.InstanceNorm2d(planes//2, momentum=BN_MOMENTUM, affine=True)
        # self.ghost_conv2 = GhostModule(inp=planes, oup=planes, kernel_size=3,
        #                                stride=stride, relu=True)
        self.s_ghost_conv2 = S_Ghost(inp=planes, oup=planes, kernel_size=3,
                                     stride=stride, relu=True)
        self.epsa = PSAModule(inplans=planes, planes=planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        # 将第一个1*1卷积输出特征按通道平分分别进行BN和LN
        out1, out2 = out.chunk(2, dim=1)
        out1 = self.bn1(out1.contiguous())
        out2 = self.ln1(out2.contiguous())
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)

        out = self.s_ghost_conv2(out)
        out = self.epsa(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


"""
以下两个模块SandGlassBasicBlock和SandGlassBottleBlock模块是借鉴了MobileNeXt中的沙漏块结构
并且分别加入了SA注意力和EPAS注意力
这两个模块先不加入模型进行训练，还是先使用FusedBlock模块和GhostBottleBlock训练完成之后看效果再决定是否使用
"""
# 借鉴MobileNeXt中的沙漏块改进的basicblock
class SandGlassBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SandGlassBasicBlock, self).__init__()

        self.reduction_ratio = 4
        hidden_dim = round(inplanes // self.reduction_ratio)
        self.identity = stride == 1 and inplanes == planes

        self.sandglass1 = nn.Sequential(
            # dw
            nn.Conv2d(in_channels=inplanes, out_channels=inplanes, kernel_size=3,
                      stride=1, padding=1, groups=inplanes, bias=False),
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.SiLU(inplace=True),
            # pw-linear
            S_Ghost(inp=inplanes, oup=hidden_dim, kernel_size=1, relu=False,
                    silu_forward=False, relu_forward=False),
            # pw
            S_Ghost(inp=hidden_dim, oup=planes,
                    kernel_size=1, relu=True,
                    relu_forward=False, silu_forward=True),
            # dw-linear
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3,
                      stride=stride, padding=1, groups=planes, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        )
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.sandglass2 = nn.Sequential(
            # dw
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3,
                      stride=1, padding=1, groups=planes, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.SiLU(inplace=True),
            # pw-linear
            S_Ghost(inp=planes, oup=hidden_dim, kernel_size=1, relu=False,
                    silu_forward=False, relu_forward=False),
            # pw
            S_Ghost(inp=hidden_dim, oup=planes,
                    kernel_size=1, relu=True,
                    relu_forward=False, silu_forward=True),
            # dw-linear
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3,
                      stride=stride, padding=1, groups=planes, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.eca = ECA()
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        if self.identity:
            out = self.sandglass1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.sandglass2(out)
            out = self.bn2(out)
            out = self.eca(out)
            out = residual + out
        else:
            out = self.sandglass1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.sandglass2(out)
            out = self.bn2(out)
            out = self.eca(out)

        out = self.relu(out)

        return out


# 借鉴MobileNeXt中的沙漏块Bottleneck
class SandGlassBottleBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SandGlassBottleBlock, self).__init__()

        self.identity = stride == 1 and inplanes == planes
        self.reduction_ratio = 4
        hidden_dim = round(inplanes // self.reduction_ratio)

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(in_channels=inplanes, out_channels=inplanes, kernel_size=3,
                      stride=1, padding=1, groups=inplanes, bias=False),
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU6(inplace=True),
            # pw-linear
            GhostModule(inp=inplanes, oup=hidden_dim, kernel_size=1, relu=False),
            # nn.Conv2d(in_channels=inplanes, out_channels=hidden_dim, kernel_size=1,
            #           stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(hidden_dim, momentum=BN_MOMENTUM),
            # pw
            GhostModule(inp=hidden_dim, oup=planes*4,
                        kernel_size=1, relu=True),
            # nn.Conv2d(in_channels=hidden_dim, out_channels=planes, kernel_size=1,
            #           stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            # nn.ReLU6(inplace=True),
            # EPAS
            PSAModule(inplans=planes*4, planes=planes*4),
            # dw-linear
            nn.Conv2d(in_channels=planes*4, out_channels=planes*4, kernel_size=3,
                      stride=stride, padding=1, groups=planes, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        )

        self.branch1 = nn.Sequential(
            # # expand conv
            GhostModule(inp=inplanes, oup=planes,
                        kernel_size=1, relu=True),
            # dw conv
            # ConvBNActivation(in_planes=planes,
            #                  out_planes=planes,
            #                  kernel_size=3,
            #                  stride=stride,
            #                  groups=planes),
            depthwise_conv(inp=planes, oup=planes, kernel_size=3,
                           stride=stride, relu=False),
            # # se
            # SqueezeExcitation(input_c=inplanes,
            #                   expand_c=planes),
            # # sa
            # SqueezeAttention(channel=planes, groups=32),
            PSAModule(inplans=planes, planes=planes),
            # # linear pw conv
            GhostModule(inp=planes, oup=planes*4, kernel_size=1, relu=False)
        )
        self.downsample = downsample

    def forward(self, x):
        global out2
        out1 = self.branch1(x)

        if self.downsample is not None:
            out2 = self.downsample(x)

        out = out1 + out2

        return out



