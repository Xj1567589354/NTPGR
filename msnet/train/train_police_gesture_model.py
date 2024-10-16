from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from pgdataset.s3_handcraft import PgdHandcraft
from constants.enum_keys import PG
from models.gesture_recognition_model import GestureRecognitionModel
from torch import optim
from constants import settings


class Trainer:
    def __init__(self, is_unittest=False):
        self.is_unittest = is_unittest          # 是否是单元测试
        self.batch_size = 1  # Not bigger than num of training videos
        self.clip_len = 15*30                   # 视频片段长度，帧数*时长
        # 加载数据集，返回骨骼长度和角度
        pgd = PgdHandcraft(Path(r'F:\PythonLearn\dataset\PoliceGestureLong'), True, (512, 512), clip_len=self.clip_len)
        self.data_loader = DataLoader(pgd, batch_size=self.batch_size, shuffle=False, num_workers=settings.num_workers)
        # 手势识别模型
        self.model = GestureRecognitionModel(batch=self.batch_size)
        self.model.train()      # 训练模式
        self.loss = CrossEntropyLoss()  # The input is expected to contain raw, unnormalized scores for each class.
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self):
        step = 0
        self.model.load_ckpt()              # 加载预训练模型
        for epoch in range(100000):
            for ges_data in self.data_loader:
                # Shape: (N,F,C) N:Batch F:Frame C:Channel(concatenated features)
                # 将手势数据拼接为特征张量
                features = torch.cat((ges_data[PG.BONE_LENGTH], ges_data[PG.BONE_ANGLE_COS],
                                      ges_data[PG.BONE_ANGLE_SIN]), dim=2)
                features = features.permute(1, 0, 2)  # NFC->FNC
                features = features.to(self.model.device, dtype=torch.float32)
                # 初始化隐藏状态和细胞状态
                h0, c0 = self.model.h0(), self.model.c0()
                # class_out: (batch, num_class)
                # 得到模型预测的分类结果以及最后一个时间步长的隐藏状态和细胞状态
                _, h, c, class_out = self.model(features, h0, c0)
                # 将手势标签转换为对应的张量格式
                target = ges_data[PG.GESTURE_LABEL]
                target = target.to(self.model.device, dtype=torch.long)
                target = target.permute(1, 0)
                # Cross Entropy, Input: (N, C), Target: (N).
                target = target.reshape((-1))  # new shape: (seq_len*batch)
                loss_tensor = self.loss(class_out, target)      # 计算损失
                self.opt.zero_grad()
                loss_tensor.backward()
                self.opt.step()     # 更新模型参数

                # 如果步长为100的倍数，则输出损失日志
                if step % 100 == 0:
                    print("Step: %d, Loss: %f" % (step, loss_tensor.item()))
                # 如果步长为5000的倍数且不为0，则保存模型权重
                if step % 5000 == 0 and step != 0:
                    self.model.save_ckpt()
                # 如果为真，完成一个epoch结束
                if self.is_unittest:
                    break
                step = step + 1
            # 如果为真，完成一个epoch结束
            if self.is_unittest:
                break
