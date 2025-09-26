import torch.nn as nn
import torch


class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log Likelihood Loss"""

    def __init__(
        self,
        ignore_value=None,
        pre_calculation_function=None,
    ):
        super().__init__()
        self.ignore_value = ignore_value
        self.pre_calculation_function = pre_calculation_function

    def forward(self, out, target):
        """
        Applies the Gaussian NLL loss
        :param out: output of the network, shape [B,2,H,W]
        :param target: target [B,1,H,W]
        :return: Gaussian NLL loss
        """
        if self.pre_calculation_function != None:
            out, target = self.pre_calculation_function(out, target)

        mean = out[:, 0, ...]
        log_var = out[:, 1, ...]

        mean = mean.flatten()
        log_var = log_var.flatten()
        target = target.flatten()

        if self.ignore_value is not None:
            mean = mean[target != self.ignore_value]
            log_var = log_var[target != self.ignore_value]
            target = target[target != self.ignore_value]

        var = torch.exp(log_var)
        epsilon = 1e-6
        loss = 0.5 * ((target - mean)**2 / (var + epsilon) + log_var)

        return loss.mean()
