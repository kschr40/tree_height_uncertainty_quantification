import torch
import torch.nn as nn
import math

class LogNormalNLLLoss(nn.Module):
    """Log-Normal Negative Log Likelihood Loss"""

    def __init__(self, ignore_value=None, pre_calculation_function=None):
        super().__init__()
        self.ignore_value = ignore_value
        self.pre_calculation_function = pre_calculation_function

    def forward(self, out, target):
        """
        Applies the Log-Normal NLL loss
        :param out: output of the network, shape [B,2,H,W]
        :param target: target [B,1,H,W]
        :return: Log-Normal NLL loss
        """
        if self.pre_calculation_function is not None:
            out, target = self.pre_calculation_function(out, target)

        LOG_SIG_MIN = -7.0
        LOG_SIG_MAX = 7.0

        epsilon = 1e-6  # Small constant to avoid log(0)
        mean = out[:, 0, ...]
        log_var = out[:, 1, ...]

        mean = mean.flatten()
        log_var = log_var.flatten()
        target = target.flatten()

        if self.ignore_value is not None:
            mean = mean[target != self.ignore_value]
            log_var = log_var[target != self.ignore_value]
            target = target[target != self.ignore_value]

        target = torch.clamp(target, min=epsilon)
        log_target = torch.log(target)
        log_var = torch.clamp(log_var, LOG_SIG_MIN, LOG_SIG_MAX)
        var = torch.exp(log_var)

        # Compute the log-likelihood
        loss = 0.5 * ((log_target - mean)**2 / var + log_var)
        # Return the negative log-likelihood
        return loss.mean()
