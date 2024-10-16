import torch
from torch import nn
import torch.nn.functional as F
from models.modules import *


def _weights_initializer():
    pass


class Aconv1d(nn.Module):
    def __init__(self, dilation, channel_in, channel_out, activate='sigmoid'):
        super(Aconv1d, self).__init__()

        assert activate in ['sigmoid', 'tanh']

        self.dilation = dilation
        self.activate = activate

        self.dilation_conv1d = nn.Conv1d(in_channels=channel_in, out_channels=channel_out,
                                       kernel_size=7, dilation=self.dilation, bias=False)
        self.bn = nn.BatchNorm1d(channel_out)

    def forward(self, inputs):
        # padding number = (kernel_size - 1) * dilation / 2
        inputs = F.pad(inputs, (3*self.dilation, 3*self.dilation))
        outputs = self.dilation_conv1d(inputs)
        outputs = self.bn(outputs)

        if self.activate == 'sigmoid':
            outputs = torch.sigmoid(outputs)
        else:
            outputs = torch.tanh(outputs)

        return outputs


# 卷积网络分支
class ResnetBlockX(nn.Module):
    def __init__(self, channel_in, channel_out, dilation):
        super(ResnetBlockX, self).__init__()
        # 因果卷积
        self.causalconv1d = nn.Conv1d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(num_features=channel_out)

        self.conv_filter = Aconv1d(dilation, channel_out, channel_out, activate='tanh')
        self.conv_gate = Aconv1d(dilation, channel_out, channel_out, activate='sigmoid')

        # ECA
        self.eca = ECoordAtt(inp=channel_out, oup=channel_out)

        # 动态卷积
        self.odconv1d = CondConv2D(channel_out, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, inputs):
        output = self.bn1(self.causalconv1d(inputs))
        output = torch.tanh(output)
        out_filter = self.conv_filter(output)
        out_gate = self.conv_gate(output)
        output = out_filter * out_gate

        output = torch.tanh(self.bn(self.odconv1d(self.eca(output))))
        out = output + inputs
        return out, output


# 三分支网络
class MSNet(nn.Module):
    def __init__(self, num_classes, channels_in, channels_out=128, num_layers=3,
                 dilations=[1, 2, 4, 8, 16], num_lstm_units=3, lstm_hidden_size=64,
                 attention_size=64, softmax=False):   # dilations=[1,2,4]
        super(MSNet, self).__init__()
        self.num_layers = num_layers
        self.bn = nn.BatchNorm1d(channels_out)
        self.softmax = softmax

        # ResnetBlockX, 15 blocks
        self.resnet_block_0 = nn.ModuleList([ResnetBlockX(dilation, channels_in, channels_out) for dilation in dilations])
        self.resnet_block_1 = nn.ModuleList([ResnetBlockX(dilation, channels_out, channels_out) for dilation in dilations])
        self.resnet_block_2 = nn.ModuleList([ResnetBlockX(dilation, channels_out, channels_out) for dilation in dilations])

        # LSTM 分支
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size=channels_in, hidden_size=lstm_hidden_size,
                                                  num_layers=1, batch_first=True, bidirectional=True) for _ in
                                          range(num_lstm_units)])
        self.fc_lstm = nn.Linear(lstm_hidden_size * 2, channels_out)

        # 混合注意力分支
        self.ham = HAM(channels_in=channels_in, attention_size=attention_size, channels_out=channels_out)

        self.odconv1d = CondConv2D(channels_out, channels_out, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels_out)

        # last
        self.fc_final = nn.Linear(channels_out, num_classes)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        # resnetblockx
        outs = 0.0      # skip connection
        for layer in self.resnet_block_0:
            x, out = layer(inputs)
            outs += out
        for layer in self.resnet_block_1:
            x, out = layer(inputs)
            outs += out
        for layer in self.resnet_block_2:
            x, out = layer(inputs)
            outs += out

        # LSTM
        lstm_inputs = inputs.transpose(1, 2)  # 转换为 [batch, seq_len, channels]
        lstm_out_total = 0
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(lstm_inputs)
            lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
            lstm_out = self.fc_lstm(lstm_out)  # 调整维度
            lstm_out_total += lstm_out  # 累加所有 LSTM 的输出

        # hybrid attention
        attention_out = self.ham(inputs)

        outputs_conv = torch.tanh(self.bn2(self.odconv1d(self.relu(outs))))
        combined_out = outputs_conv * (lstm_out_total + attention_out)

        logits = self.fc_final(self.relu(combined_out))
        output = F.softmax(logits, dim=1) if self.softmax else logits

        return output


if __name__ == '__main__':
    input = torch.rand([4, 40, 128])
    model = WaveNet(num_classes=17, channels_in=40, residul_channels=32, skip_channels=128, softmax=True)
    model.eval()
    output = model(input)
    print(output)