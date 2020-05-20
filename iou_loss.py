import torch
from torch.autograd import Function
import logging


class IoU(Function):
    """IoU coeff for individual examples"""

    def forward(self, input, target):
        n_classes = input.size(0)
        classes = range(n_classes)
        loss = torch.zeros(n_classes, dtype=torch.float, device=input.device)
        eps = 0.0001
        input = torch.sigmoid(input)
        max_index = torch.max(input, 0).indices
        #input = (input > 0.5).float()
        for class_index in classes:
            jaccard_target = (target == class_index).float()
            #jaccard_input = input[class_index, ...]
            jaccard_input = (max_index == class_index).float()

            num_preds = jaccard_target.long().sum()

            if num_preds == 0:
                loss[class_index] = 0
            else:
                inter = torch.dot(jaccard_input.flatten(), jaccard_target.flatten())
                union = torch.sum(jaccard_input) + torch.sum(jaccard_target) - inter + eps
                t = (inter.float() + eps) / union.float()
                loss[class_index] = t
        logging.info(loss)
        return torch.mean(loss)


def iou_coeff(input, target):
    """IoU coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + IoU().forward(c[0], c[1])

    return s / (i + 1)
