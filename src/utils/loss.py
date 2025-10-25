import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLoss(nn.Module):
    def __init__(self, ignore_index=255, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, targets):
        
        return self.loss_fn(preds, targets)
       
        
        
