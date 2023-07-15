import torch

class WeightedBCELoss:

    def __init__(
        self, 
        positive_weight, 
        positive_weight_frac=1,
        epsilon=1e-7
    ):
        self.weight_frac = positive_weight_frac
        self.weight = positive_weight
        self.epsilon = epsilon
        


    def __call__(self, pred, true):
        if len(pred.shape) == 4:
            pred = pred.unsqueeze(dim=0)
            true = true.unsqueeze(dim=0)

        pred = torch.clip(pred, self.epsilon, 1 - self.epsilon)

        positive = (self.weight_frac * (self.weight - 1) + 1) * true * torch.log(pred)
        negative = (1 - true) * torch.log(1 - pred)

        total = positive + negative

        loss_temp = total.sum()
        loss = (-1 / pred.shape[0]) * loss_temp

        return loss



class SoftDiceLoss:
    
    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon
        
        

    def __call__(self, pred, true):
        pred = pred.reshape(-1)
        true = true.reshape(-1)
        
        pred = torch.clip(pred, self.epsilon, 1 - self.epsilon)

        intersection = (pred * true).sum()
        dice_coefficient = (2.0 * intersection) / (pred.sum() + true.sum())

        dice_loss = 1.0 - dice_coefficient

        return dice_loss
    
    
    
class HardDiceLoss:
    
    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon
        
        

    def __call__(self, pred, true):
        pred = pred.reshape(-1)
        true = true.bool().reshape(-1)
        
        pred = pred > 0.5

        intersection = (pred & true).sum()
        dice_coefficient = (2.0 * intersection) / (pred.sum() + true.sum())

        dice_loss = 1.0 - dice_coefficient

        return dice_loss
    
    
    
class CompositeLoss:
    
    def __init__(self, 
        positive_weight, 
        wbce_positive_frac=1,
        wbce_weight=1, 
        dice_weight=100, 
        epsilon=1e-7
    ):
        self.wbce_weight = wbce_weight
        self.dice_weight = dice_weight
        
        self.wbce = WeightedBCELoss(
            positive_weight, 
            positive_weight_frac=wbce_positive_frac,
            epsilon=epsilon
        )
        self.dice = SoftDiceLoss(epsilon=epsilon)
        
    
    
    def __call__(self, pred, true):
        wbce = self.wbce_weight * self.wbce(pred, true)
        dice = self.dice_weight * self.dice(pred, true)
        
        return wbce + dice
        