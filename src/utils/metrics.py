import torch
import numpy as np


def intersection_and_union(pred, target, num_classes, ignore_index=255):
    """Compute intersection and union between prediction and ground truth."""
    # pred initial shape [B, H, W]
    
    pred = pred.view(-1)
    target = target.view(-1)

    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    intersection = pred[pred == target]
    area_intersection = torch.histc(intersection.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_pred = torch.histc(pred.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_target = torch.histc(target.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_union = area_pred + area_target - area_intersection

    return area_intersection, area_union

