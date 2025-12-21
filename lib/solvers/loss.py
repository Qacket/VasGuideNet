import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.ndimage import distance_transform_edt as edt
from monai.data import MetaTensor
import numpy as np


class CustomDiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-5

    def forward(self, input, target):
        # 转换MetaTensor为Tensor
        if isinstance(input, MetaTensor):
            input = input.as_tensor()
        if isinstance(target, MetaTensor):
            target = target.as_tensor()

        # 打印输入和目标张量的属性
        print(
            f"Input shape: {input.shape}, dtype: {input.dtype}, device: {input.device}"
        )
        print(
            f"Target shape: {target.shape}, dtype: {target.dtype}, device: {target.device}"
        )

        # 检查NaN和Inf值
        if torch.isnan(input).any() or torch.isinf(input).any():
            raise ValueError("Input tensor contains NaNs or Infs")
        if torch.isnan(target).any() or torch.isinf(target).any():
            raise ValueError("Target tensor contains NaNs or Infs")

        # 确认设备一致性
        if input.device != target.device:
            raise ValueError("Input and target tensors are on different devices")

        reduce_axis = (2, 3) if input.ndim == 4 else (1, 2, 3)
        intersection = torch.sum(target * input, dim=reduce_axis)
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (
            torch.sum(input, dim=reduce_axis)
            + torch.sum(target, dim=reduce_axis)
            + self.smooth
        )

        # 计算交叉熵损失部分
        ce_loss = nn.functional.cross_entropy(input, target)

        # 合并损失
        total_loss = dice_loss + ce_loss

        # 调用父类的forward方法
        return total_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        self.size_average = size_average
        # self.num_classes = num_classes

    def forward(self, inputs, target):
        if inputs.dim() > 2:
            inputs = inputs.reshape(
                inputs.size(0), inputs.size(1), -1
            )  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().reshape(
                -1, inputs.size(2)
            )  # N,H*W,C => N*H*W,C
        target = target.reshape(-1, 1)
        # target_one_hot = torch.zeros(target.size(0), self.num_classes).to(input.device)
        # target_one_hot.scatter_(1)
        logpt = F.log_softmax(inputs, -1)
        logpt = logpt.gather(1, target)
        logpt = logpt.reshape(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def dice_coef(y_pred, y_true, smooth=1e-8):
    intersection = (y_true * y_pred).sum()
    return (2.0 * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)


class GlobalDiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        num_classes = y_pred.shape[1]
        if y_pred.dim() > 2:
            y_pred = y_pred.reshape(
                y_pred.size(0), y_pred.size(1), -1
            )  # N,C,H,W => N,C,H*W
            y_pred = y_pred.transpose(1, 2)  # N,C,H*W => N,H*W,C
            y_pred = y_pred.contiguous().reshape(-1, y_pred.size(2))
        pred = y_pred
        target = y_true.flatten().to(torch.int64)
        target_one_hot = torch.zeros(target.size(0), num_classes).to(y_pred.device)
        target_one_hot.scatter_(1, target.unsqueeze(-1), 1)
        dice_score = dice_coef(
            torch.softmax(pred, -1)[:, 1:].flatten(), target_one_hot[:, 1:].flatten()
        )
        dice_loss = 1 - dice_score
        return dice_loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert (
            inputs.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            inputs.size(), target.size()
        )
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim < 3:  # 2D or 3D data without batch dimension
            img = img.unsqueeze(0)
        field = torch.zeros_like(img)

        for batch in range(img.size(0)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                # Use torch's distance transform equivalent if available
                fg_dist = torch.tensor(edt(fg_mask.cpu().numpy())).to(img.device)
                bg_dist = torch.tensor(edt(bg_mask.cpu().numpy())).to(img.device)

                field[batch] = fg_dist + bg_dist

        if field.shape[0] == 1:
            field = field[0]
        return field

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.ndim in {2, 3, 4, 5}, "Only 2D and 3D inputs are supported"
        assert (
            pred.ndim == target.ndim
        ), "Prediction and target need to be of same dimension"

        if pred.ndim < 4:  # 2D or 3D data
            pred = pred.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)

        pred_dt = self.distance_field(pred)
        target_dt = self.distance_field(target)

        pred_error = (pred - target) ** 2
        distance = pred_dt**self.alpha + target_dt**self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        return loss  # Directly return the loss
