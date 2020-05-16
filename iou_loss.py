import torch
from torch.autograd import Function


class IoU(Function):
    """IoU coeff for individual examples"""

    def forward(self, input, target):
        n_classes = input.size(0)
        classes = range(n_classes)
        loss = torch.zeros(n_classes, dtype=torch.float, device=input.device)
        eps = 0.0001
        for class_index in classes:
            jaccard_target = (target == class_index).float()
            jaccard_input = input[class_index, ...]

            num_preds = jaccard_target.long().sum()

            if num_preds == 0:
                loss[class_index] = 0
            else:
                self.inter = torch.dot(jaccard_input.flatten(), jaccard_target.flatten())
                self.union = torch.sum(jaccard_input) + torch.sum(jaccard_target) + eps
                t = (self.inter.float() + eps) / self.union.float()
                loss[class_index] = t

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
