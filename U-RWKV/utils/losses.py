import torch
import torch.nn as nn
import torch.nn.functional as F


# __all__ = ['BCEDiceLoss']
__all__ = ['DiceLoss','BCEDiceLoss', 'IOULoss', 'BCEIOULoss',  'IOUDiceLoss', 'BCEIOUDiceLoss']  # 添加新的 Loss 类

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice
    
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5*bce + 0.5 * dice


class IOULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1e-5

        # 将输入通过 sigmoid 激活函数，并将其转换为二值化输出
        input = torch.sigmoid(input)
        input_ = input > 0.5
        target_ = target > 0.5

        # 计算交集和并集
        intersection = (input_ & target_).sum()
        union = (input_ | target_).sum()

        # 计算 IoU
        iou = (intersection + smooth) / (union + smooth)

        # 由于我们希望最小化损失，因此使用 1 - IoU
        iou_loss = 1 - iou

        return iou_loss


class BCEIOULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # 计算 BCE 损失
        bce = F.binary_cross_entropy_with_logits(input, target)

        # 计算 IoU 损失
        iou_loss = IOULoss()(input, target)

        # 返回 0.5 * BCE + 0.5 * IoU
        return 0.5 * bce + 0.5 * iou_loss
    
class IOUDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # 计算 BCE 损失
        # bce = F.binary_cross_entropy_with_logits(input, target)

        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        # 计算 IoU 损失
        iou_loss = IOULoss()(input, target)

        # 返回 0.5 * BCE + 0.5 * IoU
        return 0.5 * dice + 0.5 * iou_loss
    
class BCEIOUDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # 计算 BCE 损失
        bce = F.binary_cross_entropy_with_logits(input, target)

        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        # 计算 IoU 损失
        iou_loss = IOULoss()(input, target)

        # 返回 0.5 * BCE + 0.5 * IoU
        return 0.33 * bce + 0.34 * dice +  0.33 * iou_loss



def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss
