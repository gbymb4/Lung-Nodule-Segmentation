import torch

from torch import nn
from torch.nn import MSELoss
from torchvision.models.resnet import resnet50, ResNet50_Weights

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
    
    
    
class PerceptualR50Loss:
    
    def __init__(self, device='cpu', epsilon=1e-7):
        self.epsilon = 1e-7
        
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone.eval()
        backbone.to(device)
        
        backbone = nn.Sequential(*list(backbone._modules.values())[:-2])
        
        for param in backbone.parameters():
            param.requires_grad = False

        self.backbone = backbone
        self.criterion = MSELoss()



    def __call__(self, pred, true):
        if len(pred.shape) == 5:
            pred = pred.squeeze(dim=0)
            true = true.squeeze(dim=0)
        
        true = torch.swapaxes(true, 0, 1)
        pred = torch.swapaxes(pred, 0, 1)
        
        pred = pred.repeat(1, 3, 1, 1)
        true = true.repeat(1, 3, 1, 1)
            
            
        pred = torch.clip(pred, self.epsilon, 1 - self.epsilon)
            
        pred_fmap = self.backbone(pred)
        true_fmap = self.backbone(true)
        
        return self.criterion(pred_fmap, true_fmap)
    
    
    
class CompositeLoss:
    
    def __init__(self, 
        positive_weight, 
        wbce_positive_frac=1,
        wbce_weight=1, 
        dice_weight=100, 
        perc_weight=1,
        epsilon=1e-7,
        device='cpu'
    ):
        self.wbce_weight = wbce_weight
        self.dice_weight = dice_weight
        self.perc_weight = perc_weight
        
        if self.wbce_weight > 0:
            self.wbce = WeightedBCELoss(
                positive_weight, 
                positive_weight_frac=wbce_positive_frac,
                epsilon=epsilon
            )
        if self.dice_weight > 0:
            self.dice = SoftDiceLoss(epsilon=epsilon)
        if self.perc_weight > 0:
            self.perceptual = PerceptualR50Loss(device=device, epsilon=epsilon)
        
    
    
    def __call__(self, pred, true):
        if self.wbce_weight > 0:
            wbce = self.wbce_weight * self.wbce(pred, true)
        else:
            wbce = 0
            
        if self.dice_weight > 0:
            dice = self.dice_weight * self.dice(pred, true)
        else:
            dice = 0
            
        if self.perc_weight > 0:
            perceptual = self.perc_weight * self.perceptual(pred, true)
        else:
            perceptual = 0
        
        return perceptual + wbce + dice
        
