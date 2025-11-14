from torch import nn
import torch
import itertools
from .shift_loss import ShiftLoss

class ShiftGaussianNLLLoss(torch.nn.Module):
    """
    Implements the pinball version of the shifted loss
    """
    def __init__(self, ignore_value=None, min_measurements=5, radius=1):
        super(ShiftGaussianNLLLoss, self).__init__()

        def loss_function(shifted_labels, predictions):
            mean_predictions = predictions[:,0:1,...]
            logvar_predictions = predictions[:,1:2,...]
            var_predictions = torch.exp(logvar_predictions)
            epsilon = 1e-6
            residuals = 0.5 * ((shifted_labels.unsqueeze(1) - mean_predictions) ** 2 / (var_predictions + epsilon) + logvar_predictions)
            # residuals = 0.5 * (logvar_predictions + ((shifted_labels - mean_predictions) ** 2) / var_predictions)
            return residuals

        self.shift_loss = ShiftLoss(loss_function, ignore_value, min_measurements, radius)

    def forward(self, predictions, labels):
        return self.shift_loss(predictions, labels)
