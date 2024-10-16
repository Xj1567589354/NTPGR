import torch
from torch import nn
from torch.nn import LSTM
from constants.keypoints import aic_bones, aic_bone_pairs
from pathlib import Path
from models.msnet import *


# 手势识别预测模型
class GestureRecognitionModel(nn.Module):
    def __init__(self, batch):
        super().__init__()
        # 输入特征的维度，关键点数目+2*关键点骨骼对数目(感觉这里和论文不太对应，这里计算的两个角度值，是骨骼配对关系之间的角度)
        num_input = len(aic_bones) + 2*len(aic_bone_pairs)
        self.num_hidden = 48        # 隐藏层大小
        self.num_output = 9         # 输出类别个数
        self.batch = batch
        self.msnet = MSNet(num_classes=self.num_output, channels_in=num_input, lstm_hidden_size=self.num_hidden)      # msnet
        self.drop = nn.Dropout(p=0.5)

        self.ckpt_path = Path(r'F:\PythonLearn\DeepLearn\ctpgr-pytorch-master\checkpoints\lstm.pt')                # 模型权重路径
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device, dtype=torch.float32)

    # 保存模型权重
    def save_ckpt(self):
        torch.save(self.state_dict(), self.ckpt_path)
        print('LSTM checkpoint saved.')

    # 加载预训练模型权重
    def load_ckpt(self, allow_new=True):
        if Path.is_file(self.ckpt_path):
            checkpoint = torch.load(self.ckpt_path)
            self.load_state_dict(checkpoint)
        else:
            if allow_new:
                print('LSTM ckpt not found.')
            else:
                raise FileNotFoundError('LSTM ckpt not found.')

    def forward(self, x):
        # output shape: (seq_len, batch, num_directions * hidden_size)
        # 将输入状态、隐藏状态和细胞状态输入到LSTM当中，返回每个时间步长的输出和最后一个时间步长的隐藏状态和细胞状态
        output = self.msnet(x)
        # 随机失活，防止过拟合
        class_out = self.drop(output)
        return output

    # 初始隐藏状态
    def h0(self):
        return torch.randn((1, self.batch, self.num_hidden), device=self.device)

    # 初始细胞状态
    def c0(self):
        return torch.randn((1, self.batch, self.num_hidden), device=self.device)