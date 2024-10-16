from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

from lib.models.block import *
from lib.models.attention_modules import *


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

"""
这个文件改进了HRNet最后的输出特征融合机制，并且是在HNRetV1的基础上进行改进
具体来说是，HRNet+多级上采样机制+ghots模块和sa模块替换basicblock模块+ghost模块和EPSA模块替换basicblock模块+Non-Local
+EPAS融入下采样当中
"""

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # self.sk = SKAttention(channel=planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.sk(out)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

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
        # 批量归一化和实例归一化
        self.bn1 = nn.BatchNorm2d(planes//2, momentum=BN_MOMENTUM)
        self.ln1 = nn.InstanceNorm2d(planes//2, momentum=BN_MOMENTUM, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
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

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, isstage4=False):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)
        self.isstage4 = isstage4
        # Non-Local attention
        # self.non_local_atte = NONLocalBlock2D(in_channels=self.num_inchannels[0]*8)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        # i是输出分支个数，j是输入分支个数
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:           # 当输入分支序号小于输出分支序号时，进行上采样
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:        # 输入分支序号和输出分支序号相同时
                    fuse_layer.append(None)
                else:               # 输入分支序号大于输出分支序号时，进行下采样
                    conv3x3s = []
                    for k in range(i-j):
                        # 黄色的conv+bn
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    # # EPSA注意力模块
                                    # PSAModule(inplans=num_outchannels_conv3x3, planes=num_outchannels_conv3x3)
                                    # 极化注意力
                                    ParallelPolarizedSelfAttention(channel=num_outchannels_conv3x3)
                                )
                            )
                        # 添加红色conv
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            # 第四阶段第四分支新增非局部注意力
            # if i == 3 and self.isstage4:
            #     x[i] = self.non_local_atte(x[i])
            # 原本的block处理
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        # stage4实现多尺度输出
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        final_inp_channels = sum(32 * [1, 2, 4, 8])

        self.final_layer = nn.Conv2d(
            in_channels=final_inp_channels,
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        # 关键点头部预测
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=final_inp_channels,
                out_channels=final_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(final_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=final_inp_channels,
                      out_channels=17,
                      kernel_size=1,
                      stride=1,
                      padding=0)
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

        # # ca注意力模块
        # self.ca0 = CoordAtt(inp=num_channels[0], oup=num_channels[0])
        # self.ca1 = CoordAtt(inp=num_channels[1], oup=num_channels[1])
        # self.ca2 = CoordAtt(inp=num_channels[2], oup=num_channels[2])
        # self.ca3 = CoordAtt(inp=num_channels[3], oup=num_channels[3])

        # # 改进的cabm注意力模块
        # self.cbam = CBAM()

        # # 空间和通道注意力模块
        # self.sam = SpatialAttentionModule()
        # self.eca = ECA(k=5)

        # # Non-Local attention
        # self.non_local_atte0 = NONLocalBlock2D(in_channels=num_channels[0])
        # self.non_local_atte1 = NONLocalBlock2D(in_channels=num_channels[1])
        # self.non_local_atte2 = NONLocalBlock2D(in_channels=num_channels[2])
        # self.non_local_atte3 = NONLocalBlock2D(in_channels=num_channels[3])

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True),
                            # 极化注意力
                            ParallelPolarizedSelfAttention(channel=outchannels)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

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

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            if num_branches == 4:
                modules.append(
                    HighResolutionModule(
                        num_branches,
                        block,
                        num_blocks,
                        num_inchannels,
                        num_channels,
                        fuse_method,
                        reset_multi_scale_output,
                        isstage4=False
                    )
                )
            else:
                modules.append(
                    HighResolutionModule(
                        num_branches,
                        block,
                        num_blocks,
                        num_inchannels,
                        num_channels,
                        fuse_method,
                        reset_multi_scale_output,
                        isstage4=False
                    )
                )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # # ca注意力
        # y_list[0] = self.ca0(y_list[0])
        # y_list[1] = self.ca1(y_list[1])
        # y_list[2] = self.ca2(y_list[2])
        # y_list[3] = self.ca3(y_list[3])

        # # 改进的cbam注意力
        # y_list[0] = self.cbam(y_list[0])
        # y_list[1] = self.cbam(y_list[1])
        # y_list[2] = self.cbam(y_list[2])
        # y_list[3] = self.cbam(y_list[3])

        # # 空间和通道注意力
        # y_list[0] = self.sam(y_list[0])
        # y_list[1] = self.sam(y_list[1])
        # y_list[2] = self.sam(y_list[2])
        # y_list[3] = self.sam(y_list[3])

        # # Non-Local attention
        # y_list[0] = self.non_local_atte0(y_list[0])
        # y_list[1] = self.non_local_atte1(y_list[1])
        # y_list[2] = self.non_local_atte2(y_list[2])
        # y_list[3] = self.non_local_atte3(y_list[3])

        # 获取第三分支的输出特征图高宽，并将第四分支的输出特征图上采样到第三分支上
        height2, width2 = y_list[2].size(2), y_list[2].size(3)
        x1 = F.interpolate(y_list[3], size=[height2, width2], mode='bicubic', align_corners=False)
        x1 = torch.cat([y_list[2], x1], dim=1)   # 384

        # 获取第二分支的输出特征图高宽，并将第三分支的输出特征图上采样到第二分支上
        height1, width1 = y_list[1].size(2), y_list[1].size(3)
        x2 = F.interpolate(x1, size=[height1, width1], mode='bicubic', align_corners=False)
        x2 = torch.cat([y_list[1], x2], dim=1)   # 448

        # 获取第一分支的输出特征图高宽，并将第二分支的输出特征图上采样到第一分支上
        height0, width0 = y_list[0].size(2), y_list[0].size(3)
        x3 = F.interpolate(x2, size=[height0, width0], mode='bicubic', align_corners=False)
        x3 = torch.cat([y_list[0], x3], dim=1)   # 480

        # # ECA
        # x = self.eca(x3)
        # x = self.head(x)

        x = self.head(x3)
        # x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net_lastfuse(cfg, is_train, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
