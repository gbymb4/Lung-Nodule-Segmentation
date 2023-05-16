import torch

from torch import nn

class WeightedBCELoss:

    def __init__(self, positive_weight):
        self.weight = positive_weight



    def __call__(self, pred, true):
        if len(pred.shape) == 4:
            pred = pred.unsqueeze(dim=0)
            true = true.unsqueeze(dim=0)

        positive = self.weight * true * torch.log(pred)
        negative = (1 - true) * torch.log(1 - pred)

        total = positive + negative

        loss_temp = total.sum()
        loss = (-1 / pred.shape[0]) * loss_temp

        return loss



class BinaryDiceLoss:

    def __call__(self, predicted, target):
        predicted = predicted.view(-1)
        target = target.view(-1)

        intersection = (predicted * target).sum()
        dice_coefficient = (2.0 * intersection) / (predicted.sum() + target.sum())

        dice_loss = 1.0 - dice_coefficient

        return dice_loss