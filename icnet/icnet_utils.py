from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

#from torch.autograd import Variable

import math
from bisect import bisect_right

class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""
    
    def __init__(self, aux_weight=0.4, ignore_index=-1):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, H, W] -> [batch, 1, H, W]
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        #return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)
        return loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight




class IterationPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, target_lr=0, max_iters=0, power=0.9, last_epoch=-1):
        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        super(IterationPolyLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        N = self.max_iters 
        T = self.last_epoch
        factor = pow(1 - T / N, self.power)
        # https://blog.csdn.net/mieleizhi0522/article/details/83113824
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]