import torch
import torch.nn as nn
import math

from lib.models.attention_modules import channel_shuffle

BN_MOMENTUM = 0.1


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,
                 relu_forward=True, silu_forward=False):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)
        self.relu_forward = relu_forward
        self.silu_forward = silu_forward

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.bn = nn.BatchNorm2d(init_channels+new_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True) if relu_forward else nn.Sequential()
        self.silu = nn.SiLU(inplace=True) if silu_forward else nn.Sequential()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = out[:, :self.oup, :, :]
        out = self.bn(out)

        if self.relu_forward:
            out = self.relu(out)
        if self.silu_forward:
            out = self.silu(out)

        return out


# 使用通道混洗实现的ghost模块
class S_Ghost(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,
                 relu_forward=True, silu_forward=False):
        super(S_Ghost, self).__init__()
        self.ghost = GhostModule(inp=inp, oup=oup, kernel_size=kernel_size, ratio=ratio,
                                 dw_size=dw_size, stride=stride, relu=relu,
                                 relu_forward=relu_forward, silu_forward=silu_forward)

    def forward(self, x):
        out = self.ghost(x)
        out = channel_shuffle(out, groups=2)

        return out
