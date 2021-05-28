import torch
import torch.nn as nn
import numpy as np

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    loss2 = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    
    return loss.mean(), loss2.mean()


def calc_iou(pred, target, t=0.5):
    pred = torch.sigmoid(pred)
    pred = pred.cpu().numpy() > 0.5
    

    intersection = np.logical_and(target.cpu().numpy(), pred)
    union = np.logical_or(target.cpu().numpy(), pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

