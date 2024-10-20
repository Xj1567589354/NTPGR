import torch.nn as nn
import math

import torch
import torch.nn.functional as F


__all__ = ["UHRNet_W18_Small", "UHRNet_W18", "UHRNet_W48"]
BN_MOMENTUM = 0.1

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        if padding == 'same':
            pad = (kernel_size - 1) // 2
        else:
            pad = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.ReLU(self.bn(self.conv(x)))

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        if padding == 'same':
            pad     = (kernel_size - 1) // 2
        else:
            pad     = padding
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)

    def forward(self, x):
        return self.bn(self.conv(x))



class UHRNet(nn.Module):
    """
    The UHRNet implementation based on PaddlePaddle.
    The original article refers to
    Jian Wang, et, al. "U-HRNet: Delving into Improving Semantic Representation of High Resolution Network for Dense Prediction"
    (https://arxiv.org/pdf/2210.07140.pdf).
    Args:
        in_channels (int, optional): The channels of input image. Default: 3.
        pretrained (str): The path of pretrained model.
        stage1_num_modules (int): Number of modules for stage1. Default 1.
        stage1_num_blocks (list): Number of blocks per module for stage1. Default [4].
        stage1_num_channels (list): Number of channels per branch for stage1. Default [64].
        stage2_num_modules (int): Number of modules for stage2. Default 1.
        stage2_num_blocks (list): Number of blocks per module for stage2. Default [4, 4]
        stage2_num_channels (list): Number of channels per branch for stage2. Default [18, 36].
        stage3_num_modules (int): Number of modules for stage3. Default 5.
        stage3_num_blocks (list): Number of blocks per module for stage3. Default [4, 4]
        stage3_num_channels (list): Number of channels per branch for stage3. Default [36, 72].
        stage4_num_modules (int): Number of modules for stage4. Default 2.
        stage4_num_blocks (list): Number of blocks per module for stage4. Default [4, 4]
        stage4_num_channels (list): Number of channels per branch for stage4. Default [72. 144].
        stage5_num_modules (int): Number of modules for stage5. Default 2.
        stage5_num_blocks (list): Number of blocks per module for stage5. Default [4, 4]
        stage5_num_channels (list): Number of channels per branch for stage5. Default [144, 288].
        stage6_num_modules (int): Number of modules for stage6. Default 1.
        stage6_num_blocks (list): Number of blocks per module for stage6. Default [4, 4]
        stage6_num_channels (list): Number of channels per branch for stage6. Default [72. 144].
        stage7_num_modules (int): Number of modules for stage7. Default 1.
        stage7_num_blocks (list): Number of blocks per module for stage7. Default [4, 4]
        stage7_num_channels (list): Number of channels per branch for stage7. Default [36, 72].
        stage8_num_modules (int): Number of modules for stage8. Default 1.
        stage8_num_blocks (list): Number of blocks per module for stage8. Default [4, 4]
        stage8_num_channels (list): Number of channels per branch for stage8. Default [18, 36].
        stage9_num_modules (int): Number of modules for stage9. Default 1.
        stage9_num_blocks (list): Number of blocks per module for stage9. Default [4]
        stage9_num_channels (list): Number of channels per branch for stage9. Default [18].
        has_se (bool): Whether to use Squeeze-and-Excitation module. Default False.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 pretrained=None,
                 stage1_num_modules=1,
                 stage1_num_blocks=(4, ),
                 stage1_num_channels=(64, ),
                 stage2_num_modules=1,
                 stage2_num_blocks=(4, 4),
                 stage2_num_channels=(18, 36),
                 stage3_num_modules=5,
                 stage3_num_blocks=(4, 4),
                 stage3_num_channels=(36, 72),
                 stage4_num_modules=2,
                 stage4_num_blocks=(4, 4),
                 stage4_num_channels=(72, 144),
                 stage5_num_modules=2,
                 stage5_num_blocks=(4, 4),
                 stage5_num_channels=(144, 288),
                 stage6_num_modules=1,
                 stage6_num_blocks=(4, 4),
                 stage6_num_channels=(72, 144),
                 stage7_num_modules=1,
                 stage7_num_blocks=(4, 4),
                 stage7_num_channels=(36, 72),
                 stage8_num_modules=1,
                 stage8_num_blocks=(4, 4),
                 stage8_num_channels=(18, 36),
                 stage9_num_modules=1,
                 stage9_num_blocks=(4, ),
                 stage9_num_channels=(18, ),
                 has_se=False,
                 align_corners=False):
        super(UHRNet, self).__init__()
        self.inplanes = 64
        self.has_se = has_se
        self.align_corners = align_corners
        self.feat_channels = [
            sum([
                stage5_num_channels[-1], stage6_num_channels[-1],
                stage7_num_channels[-1], stage8_num_channels[-1],
                stage9_num_channels[-1]
            ]) // 2
        ]

        cur_stride = 1
        # stem net
        # self.conv_layer1_1 = ConvBnReLU(
        #     in_channels=3,
        #     out_channels=64,
        #     kernel_size=3,
        #     stride=2,
        #     padding='same')
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        cur_stride *= 2

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        cur_stride *= 2

        self.layer1 = Layer1(
            num_channels=64,
            num_blocks=stage1_num_blocks[0],
            num_filters=stage1_num_channels[0],
            has_se=has_se,
            name="layer2")

        self.transition1 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage1_num_channels[0] * 4,
            stride_cur=[
                cur_stride * (2**i) for i in range(len(stage2_num_channels))
            ],
            out_channels=stage2_num_channels,
            align_corners=self.align_corners,
            name="tr1")
        self.stage2 = Stage(
            num_channels=stage2_num_channels,
            num_modules=stage2_num_modules,
            num_blocks=stage2_num_blocks,
            num_filters=stage2_num_channels,
            has_se=self.has_se,
            name="st2",
            align_corners=align_corners)
        cur_stride *= 2

        self.transition2 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage2_num_channels[-1],
            stride_cur=[
                cur_stride * (2**i) for i in range(len(stage3_num_channels))
            ],
            out_channels=stage3_num_channels,
            align_corners=self.align_corners,
            name="tr2")
        self.stage3 = Stage(
            num_channels=stage3_num_channels,
            num_modules=stage3_num_modules,
            num_blocks=stage3_num_blocks,
            num_filters=stage3_num_channels,
            has_se=self.has_se,
            name="st3",
            align_corners=align_corners)
        cur_stride *= 2

        self.transition3 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage3_num_channels[-1],
            stride_cur=[
                cur_stride * (2**i) for i in range(len(stage4_num_channels))
            ],
            out_channels=stage4_num_channels,
            align_corners=self.align_corners,
            name="tr3")
        self.stage4 = Stage(
            num_channels=stage4_num_channels,
            num_modules=stage4_num_modules,
            num_blocks=stage4_num_blocks,
            num_filters=stage4_num_channels,
            has_se=self.has_se,
            name="st4",
            align_corners=align_corners)
        cur_stride *= 2

        self.tr4 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage4_num_channels[-1],
            stride_cur=[
                cur_stride * (2**i) for i in range(len(stage5_num_channels))
            ],
            out_channels=stage5_num_channels,
            align_corners=self.align_corners,
            name="tr4")
        self.st5 = Stage(
            num_channels=stage5_num_channels,
            num_modules=stage5_num_modules,
            num_blocks=stage5_num_blocks,
            num_filters=stage5_num_channels,
            has_se=self.has_se,
            name="st5",
            align_corners=align_corners)

        self.tr5 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage5_num_channels[0],
            stride_cur=[
                cur_stride // (2**(len(stage6_num_channels) - i - 1))
                for i in range(len(stage6_num_channels))
            ],
            out_channels=stage6_num_channels,
            align_corners=self.align_corners,
            name="tr5")
        self.st6 = Stage(
            num_channels=stage6_num_channels,
            num_modules=stage6_num_modules,
            num_blocks=stage6_num_blocks,
            num_filters=stage6_num_channels,
            has_se=self.has_se,
            name="st6",
            align_corners=align_corners)
        cur_stride = cur_stride // 2

        self.tr6 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage6_num_channels[0],
            stride_cur=[
                cur_stride // (2**(len(stage7_num_channels) - i - 1))
                for i in range(len(stage7_num_channels))
            ],
            out_channels=stage7_num_channels,
            align_corners=self.align_corners,
            name="tr6")
        self.st7 = Stage(
            num_channels=stage7_num_channels,
            num_modules=stage7_num_modules,
            num_blocks=stage7_num_blocks,
            num_filters=stage7_num_channels,
            has_se=self.has_se,
            name="st7",
            align_corners=align_corners)
        cur_stride = cur_stride // 2

        self.tr7 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage7_num_channels[0],
            stride_cur=[
                cur_stride // (2**(len(stage8_num_channels) - i - 1))
                for i in range(len(stage8_num_channels))
            ],
            out_channels=stage8_num_channels,
            align_corners=self.align_corners,
            name="tr7")
        self.st8 = Stage(
            num_channels=stage8_num_channels,
            num_modules=stage8_num_modules,
            num_blocks=stage8_num_blocks,
            num_filters=stage8_num_channels,
            has_se=self.has_se,
            name="st8",
            align_corners=align_corners)
        cur_stride = cur_stride // 2

        self.tr8 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage8_num_channels[0],
            stride_cur=[
                cur_stride // (2**(len(stage9_num_channels) - i - 1))
                for i in range(len(stage9_num_channels))
            ],
            out_channels=stage9_num_channels,
            align_corners=self.align_corners,
            name="tr8")
        self.st9 = Stage(
            num_channels=stage9_num_channels,
            num_modules=stage9_num_modules,
            num_blocks=stage9_num_blocks,
            num_filters=stage9_num_channels,
            has_se=self.has_se,
            name="st9",
            align_corners=align_corners)

        self.last_layer = nn.Sequential(
            ConvBnReLU(
                in_channels=self.feat_channels[0],
                out_channels=self.feat_channels[0],
                kernel_size=1,
                padding='same',
                stride=1,
                bias=True),
            nn.Conv2d(
                in_channels=self.feat_channels[0],
                out_channels=19,
                kernel_size=1,
                stride=1,
                padding=0))

    def _concat(self, x1, x2):
        x1 = F.avg_pool3d(
            x1.unsqueeze(1), kernel_size=(2, 1, 1), stride=(2, 1, 1)).squeeze(1)
        x2 = F.avg_pool3d(
            x2.unsqueeze(1), kernel_size=(2, 1, 1), stride=(2, 1, 1)).squeeze(1)
        return torch.concat([x1, x2], axis=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # conv1 = self.conv_layer1_1(x)
        # conv2 = self.conv_layer1_2(conv1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        conv2 = self.relu(x)

        la1 = self.layer1(conv2)

        tr1 = self.transition1(la1)
        st2 = self.stage2(tr1)
        skip21 = st2[0]

        tr2 = self.transition2(st2[-1])
        st3 = self.stage3(tr2)
        skip31 = st3[0]

        tr3 = self.transition3(st3[-1])
        st4 = self.stage4(tr3)
        skip41 = st4[0]

        tr4 = self.tr4(st4[-1])
        st5 = self.st5(tr4)
        x5 = st5[-1]

        tr5 = self.tr5(st5[0], shape=skip41.shape[-2:])
        tr5[0] = self._concat(tr5[0], skip41)
        st6 = self.st6(tr5)
        x4 = st6[-1]

        tr6 = self.tr6(st6[0], shape=skip31.shape[-2:])
        tr6[0] = self._concat(tr6[0], skip31)
        st7 = self.st7(tr6)
        x3 = st7[-1]

        tr7 = self.tr7(st7[0], shape=skip21.shape[-2:])
        tr7[0] = self._concat(tr7[0], skip21)
        st8 = self.st8(tr7)
        x2 = st8[-1]

        tr8 = self.tr8(st8[0])
        st9 = self.st9(tr8)
        x1 = st9[-1]

        x = [x1, x2, x3, x4, x5]
        for i in range(len(x)):
            x[i] = F.avg_pool3d(
                x[i].unsqueeze(1), kernel_size=(2, 1, 1), stride=(2, 1,
                                                                  1)).squeeze(1)

        # upsampling
        x0_h, x0_w = (x[0]).shape[-2:]
        for i in range(1, len(x)):
            x[i] = F.interpolate(
                x[i],
                size=[x0_h, x0_w],
                mode='bilinear',
                align_corners=self.align_corners)
        x = torch.concat(x, axis=1)

        return x


class Layer1(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 num_blocks,
                 has_se=False,
                 name=None):
        super(Layer1, self).__init__()

        self.bottleneck_block_list = nn.Sequential()

        for i in range(num_blocks):
            self.bottleneck_block_list.add_module(
                "bb_{}_{}".format(name, i + 1),
                Bottleneck(
                    num_channels=num_channels if i == 0 else num_filters * 4,
                    num_filters=num_filters,
                    has_se=has_se,
                    stride=1,
                    downsample=True if i == 0 else False,
                    name=name + '_' + str(i + 1)))

    def forward(self, x):
        conv = x
        for block_func in self.bottleneck_block_list:
            conv = block_func(conv)
        return conv


class TransitionLayer(nn.Module):
    def __init__(self,
                 stride_pre,
                 in_channel,
                 stride_cur,
                 out_channels,
                 align_corners=False,
                 name=None):
        super(TransitionLayer, self).__init__()
        self.align_corners = align_corners
        num_out = len(out_channels)
        if num_out != len(stride_cur):
            raise ValueError(
                'The length of `out_channels` does not equal to the length of `stride_cur`'
                .format(num_out, len(stride_cur)))
        self.conv_bn_func_list = nn.ModuleList()
        for i in range(num_out):
            residual = nn.Sequential()
            if stride_cur[i] == stride_pre:
                if in_channel != out_channels[i]:
                    residual.add_module(
                        "transition_{}_layer_{}".format(name, i + 1),
                        ConvBnReLU(
                            in_channels=in_channel,
                            out_channels=out_channels[i],
                            kernel_size=3,
                            padding='same',
                            ))
                else:
                    residual = None
            elif stride_cur[i] > stride_pre:
                residual.add_module(
                    "transition_{}_layer_{}".format(name, i + 1),
                    ConvBnReLU(
                        in_channels=in_channel,
                        out_channels=out_channels[i],
                        kernel_size=3,
                        stride=2,
                        padding='same',
                        ))
            else:
                residual.add_module(
                    "transition_{}_layer_{}".format(name, i + 1),
                    ConvBnReLU(
                        in_channels=in_channel,
                        out_channels=out_channels[i],
                        kernel_size=1,
                        stride=1,
                        padding='same',
                        ))
            self.conv_bn_func_list.append(residual)

    def forward(self, x, shape=None):
        outs = []
        for conv_bn_func in self.conv_bn_func_list:
            if conv_bn_func is None:
                outs.append(x)
            else:
                out = conv_bn_func(x)
                if shape is not None:
                    out = F.interpolate(
                        out,
                        shape,
                        mode='bilinear',
                        align_corners=self.align_corners)
                outs.append(out)
        return outs


class Branches(nn.Module):
    def __init__(self,
                 num_blocks,
                 in_channels,
                 out_channels,
                 has_se=False,
                 name=None):
        super(Branches, self).__init__()

        self.basic_block_list = nn.ModuleList()

        for i in range(len(out_channels)):
            basic_block_func = nn.ModuleList()
            for j in range(num_blocks[i]):
                in_ch = in_channels[i] if j == 0 else out_channels[i]
                basic_block_func.add_module(
                    "bb_{}_branch_layer_{}_{}".format(name, i + 1, j + 1),
                    BasicBlock(
                        num_channels=in_ch,
                        num_filters=out_channels[i],
                        has_se=has_se,
                        name=name + '_branch_layer_' + str(i + 1) + '_' +
                        str(j + 1)))
            self.basic_block_list.append(basic_block_func)

    def forward(self, x):
        outs = []
        for idx, input in enumerate(x):
            conv = input
            for basic_block_func in self.basic_block_list[idx]:
                conv = basic_block_func(conv)
            outs.append(conv)
        return outs


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 num_channels,
                 num_filters,
                 has_se=False,
                 stride=1,
                 downsample=False,
                 name=None):
        super(Bottleneck, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBnReLU(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=1,
            padding='same',
            )

        self.conv2 = ConvBnReLU(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            padding='same',
            )

        self.conv3 = ConvBn(
            in_channels=num_filters,
            out_channels=num_filters * 4,
            kernel_size=1,
            padding='same',
            )

        if self.downsample:
            self.conv_down = ConvBn(
                in_channels=num_channels,
                out_channels=num_filters * 4,
                kernel_size=1,
                padding='same',
                )

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters * 4,
                num_filters=num_filters * 4,
                reduction_ratio=16,
                name=name + '_fc')

    def forward(self, x):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        if self.downsample:
            residual = self.conv_down(x)

        if self.has_se:
            conv3 = self.se(conv3)

        y = conv3 + residual
        y = F.relu(y)
        return y



class BasicBlock(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride=1,
                 has_se=False,
                 downsample=False,
                 name=None):
        super(BasicBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBnReLU(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            padding='same',
            )
        self.conv2 = ConvBn(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding='same',
            )

        if self.downsample:
            self.conv_down = ConvBnReLU(
                in_channels=num_channels,
                out_channels=num_filters,
                kernel_size=1,
                padding='same',
                )

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters,
                num_filters=num_filters,
                reduction_ratio=16,
                name=name + '_fc')

    def forward(self, x):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        if self.downsample:
            residual = self.conv_down(x)

        if self.has_se:
            conv2 = self.se(conv2)

        y = conv2 + residual
        y = F.relu(y)
        return y


class SELayer(nn.Module):
    def __init__(self, num_channels, num_filters, reduction_ratio, name=None):
        super(SELayer, self).__init__()

        self.pool2d_gap = nn.AdaptiveAvgPool2d(1)

        self._num_channels = num_channels

        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = nn.Linear(
            num_channels,
            med_ch,
            act="relu",
            param_attr=torch.ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv)))

        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = nn.Linear(
            med_ch,
            num_filters,
            act="sigmoid",
            param_attr=torch.ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv)))

    def forward(self, x):
        pool = self.pool2d_gap(x)
        pool = torch.reshape(pool, shape=[-1, self._num_channels])
        squeeze = self.squeeze(pool)
        excitation = self.excitation(squeeze)
        excitation = torch.reshape(
            excitation, shape=[-1, self._num_channels, 1, 1])
        out = x * excitation
        return out


class Stage(nn.Module):
    def __init__(self,
                 num_channels,
                 num_modules,
                 num_blocks,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False):
        super(Stage, self).__init__()

        self._num_modules = num_modules

        self.stage_func_list = nn.Sequential()
        for i in range(num_modules):
            if i == num_modules - 1 and not multi_scale_output:
                self.stage_func_list.add_module(
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                        has_se=has_se,
                        multi_scale_output=False,
                        name=name + '_' + str(i + 1),
                        align_corners=align_corners))
            else:
                self.stage_func_list.add_module(
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                        has_se=has_se,
                        name=name + '_' + str(i + 1),
                        align_corners=align_corners))

    def forward(self, x):
        out = x
        for idx in range(self._num_modules):
            out = self.stage_func_list[idx](out)
        return out


class HighResolutionModule(nn.Module):
    def __init__(self,
                 num_channels,
                 num_blocks,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False):
        super(HighResolutionModule, self).__init__()

        self.branches_func = Branches(
            num_blocks=num_blocks,
            in_channels=num_channels,
            out_channels=num_filters,
            has_se=has_se,
            name=name)

        self.fuse_func = FuseLayers(
            in_channels=num_filters,
            out_channels=num_filters,
            multi_scale_output=multi_scale_output,
            name=name,
            align_corners=align_corners)

    def forward(self, x):
        out = self.branches_func(x)
        out = self.fuse_func(out)
        return out


class FuseLayers(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False):
        super(FuseLayers, self).__init__()

        self._actual_ch = len(in_channels) if multi_scale_output else 1
        self._in_channels = in_channels
        self.align_corners = align_corners

        self.residual_func_list = nn.Sequential()
        for i in range(self._actual_ch):
            for j in range(len(in_channels)):
                if j > i:
                    self.residual_func_list.add_module(
                        "residual_{}_layer_{}_{}".format(name, i + 1, j + 1),
                        ConvBn(
                            in_channels=in_channels[j],
                            out_channels=out_channels[i],
                            kernel_size=1,
                            padding='same',
                            ))
                elif j < i:
                    pre_num_filters = in_channels[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            self.residual_func_list.add_module(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvBn(
                                    in_channels=pre_num_filters,
                                    out_channels=out_channels[i],
                                    kernel_size=3,
                                    stride=2,
                                    padding='same',
                                    ))
                            pre_num_filters = out_channels[i]
                        else:
                            self.residual_func_list.add_module(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvBnReLU(
                                    in_channels=pre_num_filters,
                                    out_channels=out_channels[j],
                                    kernel_size=3,
                                    stride=2,
                                    padding='same',
                                    ))
                            pre_num_filters = out_channels[j]

        if len(self.residual_func_list) == 0:
            self.residual_func_list.add_module("identity",
                                  nn.Identity())  # for flops calculation

    def forward(self, x):
        outs = []
        residual_func_idx = 0
        for i in range(self._actual_ch):
            residual = x[i]
            residual_shape = residual.shape[-2:]

            for j in range(len(self._in_channels)):
                if j > i:
                    y = self.residual_func_list[residual_func_idx](x[j])
                    residual_func_idx += 1

                    y = F.interpolate(
                        y,
                        residual_shape,
                        mode='bilinear',
                        align_corners=self.align_corners)
                    residual = residual + y
                elif j < i:
                    y = x[j]
                    for k in range(i - j):
                        y = self.residual_func_list[residual_func_idx](y)
                        residual_func_idx += 1

                    residual = residual + y

            residual = F.relu(residual)
            outs.append(residual)

        return outs


def UHRNet_W18_Small(**kwargs):
    model = UHRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[2],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[2, 2],
        stage2_num_channels=[18, 36],
        stage3_num_modules=2,
        stage3_num_blocks=[2, 2],
        stage3_num_channels=[36, 72],
        stage4_num_modules=2,
        stage4_num_blocks=[2, 2],
        stage4_num_channels=[72, 144],
        stage5_num_modules=2,
        stage5_num_blocks=[2, 2],
        stage5_num_channels=[144, 288],
        stage6_num_modules=1,
        stage6_num_blocks=[2, 2],
        stage6_num_channels=[72, 144],
        stage7_num_modules=1,
        stage7_num_blocks=[2, 2],
        stage7_num_channels=[36, 72],
        stage8_num_modules=1,
        stage8_num_blocks=[2, 2],
        stage8_num_channels=[18, 36],
        stage9_num_modules=1,
        stage9_num_blocks=[2],
        stage9_num_channels=[18],
        **kwargs)
    return model


def UHRNet_W18(**kwargs):
    model = UHRNet(
        stage1_num_modules=1,
        stage1_num_blocks=(4, ),
        stage1_num_channels=(64, ),
        stage2_num_modules=1,
        stage2_num_blocks=(4, 4),
        stage2_num_channels=(18, 36),
        stage3_num_modules=5,
        stage3_num_blocks=(4, 4),
        stage3_num_channels=(36, 72),
        stage4_num_modules=2,
        stage4_num_blocks=(4, 4),
        stage4_num_channels=(72, 144),
        stage5_num_modules=2,
        stage5_num_blocks=(4, 4),
        stage5_num_channels=(144, 288),
        stage6_num_modules=1,
        stage6_num_blocks=(4, 4),
        stage6_num_channels=(72, 144),
        stage7_num_modules=1,
        stage7_num_blocks=(4, 4),
        stage7_num_channels=(36, 72),
        stage8_num_modules=1,
        stage8_num_blocks=(4, 4),
        stage8_num_channels=(18, 36),
        stage9_num_modules=1,
        stage9_num_blocks=(4, ),
        stage9_num_channels=(18, ),
        **kwargs)
    return model


def UHRNet_W48(**kwargs):
    model = UHRNet(
        stage1_num_modules=1,
        stage1_num_blocks=(4, ),
        stage1_num_channels=(64, ),
        stage2_num_modules=1,
        stage2_num_blocks=(4, 4),
        stage2_num_channels=(48, 96),
        stage3_num_modules=5,
        stage3_num_blocks=(4, 4),
        stage3_num_channels=(96, 192),
        stage4_num_modules=2,
        stage4_num_blocks=(4, 4),
        stage4_num_channels=(192, 384),
        stage5_num_modules=2,
        stage5_num_blocks=(4, 4),
        stage5_num_channels=(384, 768),
        stage6_num_modules=1,
        stage6_num_blocks=(4, 4),
        stage6_num_channels=(192, 384),
        stage7_num_modules=1,
        stage7_num_blocks=(4, 4),
        stage7_num_channels=(96, 192),
        stage8_num_modules=1,
        stage8_num_blocks=(4, 4),
        stage8_num_channels=(48, 96),
        stage9_num_modules=1,
        stage9_num_blocks=(4, ),
        stage9_num_channels=(48, ),
        **kwargs)
    return model