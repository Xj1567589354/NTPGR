import torch
import os
import logging
import torch.nn as nn

from lib.models.uhrnet_backbone import ConvBnReLU

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

"""
这个文件是U-HRNet网络结构
"""


class UHRNet(nn.Module):
    def __init__(self, cfg, num_joints=17, backbone='UHRNet_W18_Small'):
        super(UHRNet, self).__init__()
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        self.pretrained_layers = extra['PRETRAINED_LAYERS']
        if backbone == 'UHRNet_W18_Small':
            from lib.models.uhrnet_backbone import UHRNet_W18_Small
            self.backbone = UHRNet_W18_Small()
            last_inp_channels = int(279)

        if backbone == 'UHRNet_W18':
            from lib.models.uhrnet_backbone import UHRNet_W18
            self.backbone = UHRNet_W18()
            last_inp_channels = int(744)

        self.head = nn.Sequential()
        self.head.add_module(
            'conv_1', ConvBnReLU(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1,
                                 stride=1, padding=0, bias=True)
        )
        self.head.add_module(
            'last', nn.Conv2d(in_channels=last_inp_channels, out_channels=num_joints, kernel_size=1,
                              stride=1, padding=0)
        )

    def forward(self, inputs):
        # H, W = inputs.size(2), inputs.size(3)
        x = self.backbone(inputs)                           # U-HRNet backbone处理
        x = self.head(x)                                    # 关键点预测头预测
        # x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
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


def get_pose_uhrnet(cfg, is_train, **kwargs):
    model = UHRNet(cfg)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model


if __name__ == '__main__':
    input = torch.randn(8, 3, 256, 192)
    model = UHRNet()
    output = model(input)
    print(output)

