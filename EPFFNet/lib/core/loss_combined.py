from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch import Tensor


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        p = torch.sigmoid(input)
        # ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        ce_loss = self.bce(input, target)
        p_t = p * target + (1 - p) * (1 * target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 * target)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        n = target.size(0)
        input_flat = input.view(n, -1)
        target_flat = target.view(n, -1)
        intersection = input_flat * target_flat
        dice_coeff = (2. * intersection.sum(1) + self.smooth) / (input_flat.sum(1) + target_flat.sum(1) + self.smooth)
        loss = 1 - dice_coeff.sum() / n
        return loss


# 骨骼损失
class JointBoneLoss(nn.Module):
    def __init__(self, joint_num):
        super(JointBoneLoss, self).__init__()
        id_i, id_j = [], []
        for i in range(joint_num):
            for j in range(i + 1, joint_num):
                id_i.append(i)
                id_j.append(j)
        self.id_i = id_i
        self.id_j = id_j

    def forward(self, joint_out, joint_gt):
        if len(joint_out.shape) == 4:  # (b, n, h, w) heatmap-based featuremap
            calc_dim = [2, 3]
        elif len(joint_out.shape) == 3:  # (b, n, 2) or (b, n, 3) regression-based results
            calc_dim = -1

        J = torch.norm(joint_out[:, self.id_i, :] - joint_out[:, self.id_j, :], p=2, dim=calc_dim, keepdim=False)
        Y = torch.norm(joint_gt[:, self.id_i, :] - joint_gt[:, self.id_j, :], p=2, dim=calc_dim, keepdim=False)
        loss = torch.abs(J - Y)
        return loss.mean()


"""
这里使用的是MSE损失和Focal损失相结合的联合损失进行监督
原本打算是使用Focal损失和Dice损失相结合的联合损失，但是Dice损失出现问题，导致很多Nan的情况，后序还得继续研究
"""
class JointsCombinedLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsCombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(reduction='mean')
        # self.dice_loss = DiceLoss()
        # self.bone_loss = JointBoneLoss(joint_num=17)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss1 = 0
        loss2 = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss1 += 0.5 * self.focal_loss(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
                loss2 += 0.5 * self.mse_loss(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss1 += 0.5 * self.focal_loss(heatmap_pred, heatmap_gt)
                loss2 += 0.5 * self.mse_loss(heatmap_pred, heatmap_gt)

        loss1 = loss1 / num_joints
        loss2 = loss2 / num_joints

        # 联合损失
        combined_loss = loss1 + 0.25 * loss2
        return combined_loss


"""
这里使用的是MSE损失和骨骼损失相结合的联合损失进行监督
这里源代码说明用于其姿态估计任务，效果明显，后序还得做实验看看效果
"""
class JointsCombinedLoss2(nn.Module):
    def __init__(self, use_target_weight, a=0.2):
        super(JointsCombinedLoss2, self).__init__()
        self.bone_loss = JointBoneLoss(joint_num=17)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.a = a

    def forward(self, output, target, target_weight):
        batch_size, num_joints, h, w = output.size()
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss1 = 0       # 损失1
        loss2 = 0       # 损失2
        output_pred = []
        target_gt = []

        # 计算每个关键点的MSE loss
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss2 += 0.5 * self.mse_loss(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
                # 这一步骤照主要是使用target weights
                output_pred.append(heatmap_pred.mul(target_weight[:, idx]).unsqueeze(dim=1))
                target_gt.append(heatmap_gt.mul(target_weight[:, idx]).unsqueeze(dim=1))
            else:
                loss2 += 0.5 * self.mse_loss(heatmap_pred, heatmap_gt)

        # 对维度
        output_pred = torch.cat(output_pred, dim=1)
        target_gt = torch.cat(target_gt, dim=1)
        output_pred = output_pred.view(batch_size, num_joints, h, w)
        target_gt = target_gt.view(batch_size, num_joints, h, w)

        # 计算每个关键点的骨骼loss
        loss1 += 0.5 * self.bone_loss(output_pred, target_gt)

        loss1 = loss1 / num_joints
        loss2 = loss2 / num_joints

        # 联合损失，这里loss2是MSE损失，loss1是骨骼损失，骨骼损失有个权重值，默认是0.05
        combined_loss = loss2 + self.a * loss1
        return combined_loss
