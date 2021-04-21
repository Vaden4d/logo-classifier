import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """Cross Entropy with Label Smoothing"""
    def __init__(self, num_classes, smoothing=0.0, dim=-1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim
        self.weight = weight

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            if target.dim() == 1:
                smoothed = torch.zeros_like(pred)
                smoothed.fill_(self.smoothing / (self.num_classes-1))
                smoothed.scatter_(1, target.unsqueeze(1), self.confidence)
            else:
                smoothed = target * self.confidence + (1-target) * self.smoothing / (self.num_classes-1)
        if self.weight is not None:
            loss = -torch.mean((smoothed * pred) @ self.weight.to(pred.device))
        else:
            loss = -torch.mean(torch.sum(smoothed * pred, dim=self.dim))
        return loss


