from torch import nn
import torch
import itertools
from .shift_loss import ShiftLoss

class ShiftPinballLoss(torch.nn.Module):
    """
    Implements the pinball version of the shifted loss
    """
    def __init__(self, ignore_value=None, min_measurements=5, radius=1, quantiles = [0.5,0.1,0.9]):
        super(ShiftPinballLoss, self).__init__()
        self.quantiles = quantiles

        def loss_function(shifted_labels, predictions):
            residuals = torch.zeros_like(predictions).to(predictions.device)
            for i, quant in enumerate(self.quantiles):
                quant_pred = predictions[:,i,...]
                current_residuals = torch.max((quant-1) * (shifted_labels - quant_pred), quant * (shifted_labels - quant_pred))
                residuals[:,i,...] = current_residuals
            return residuals


        self.shift_loss = ShiftLoss(loss_function, ignore_value, min_measurements, radius)

    def forward(self, predictions, labels):
        return self.shift_loss(predictions, labels)
