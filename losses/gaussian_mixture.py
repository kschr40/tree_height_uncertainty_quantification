import torch.nn as nn
import torch


class GaussianMixtureLoss(nn.Module):
    """Mean Absolute error"""

    def __init__(
        self,
        ignore_value=None,
        pre_calculation_function=None
    ):
        super().__init__()
        self.ignore_value = ignore_value
        self.pre_calculation_function = pre_calculation_function

    def forward(self, out, target):
        """
        Applies the L1 loss
        :param out: output of the network
        :param target: target
        :return: l1 loss
        """
        if self.pre_calculation_function != None:
            out, target = self.pre_calculation_function(out, target)

        avg_mean = out[:, 0, ...].flatten()
        diff_avg_mean = out[:, 1, ...].flatten()

        mean0 = avg_mean - diff_avg_mean
        mean1 = avg_mean + diff_avg_mean
        log_var0 = out[:, 2, ...].flatten()
        log_var1 = out[:, 3, ...].flatten()

        target = target.flatten()

        if self.ignore_value is not None:
            mean0 = mean0[target != self.ignore_value]
            mean1 = mean1[target != self.ignore_value]
            log_var0 = log_var0[target != self.ignore_value]
            log_var1 = log_var1[target != self.ignore_value]
            target = target[target != self.ignore_value]


        var0 = torch.exp(log_var0)
        var1 = torch.exp(log_var1)
        epsilon = 1e-6
        loss0 = 0.5 * ((target - mean0)**2 / (var0 + epsilon) + log_var0)
        loss1 = 0.5 * ((target - mean1)**2 / (var1 + epsilon) + log_var1)

        return (loss0 + loss1).mean()
