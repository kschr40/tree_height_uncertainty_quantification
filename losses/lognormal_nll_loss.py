import torch
import torch.nn as nn

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

        epsilon = 1e-6  # Small constant to avoid log(0)
        exp_mu = out[:, 0, ...]
        log_var = out[:, 1, ...]

        exp_mu = exp_mu.flatten()
        log_var = log_var.flatten()
        target = target.flatten()

        if self.ignore_value is not None:
            exp_mu = exp_mu[target != self.ignore_value]
            log_var = log_var[target != self.ignore_value]
            target = target[target != self.ignore_value]


        mu = torch.log(torch.relu(exp_mu) + epsilon)
        var = torch.exp(log_var)

        log_target = torch.log(target + epsilon)  # log(target)

        # Compute the log-likelihood
        log_likelihood = -0.5 * torch.log(2 * torch.pi * var * target + epsilon) - 0.5 * ((log_target - mu) ** 2 / (var + epsilon))

        # Return the negative log-likelihood
        return -log_likelihood.mean()

# Example usage
# out: Tensor of shape [B, 2, H, W] where out[:, 0, ...] is the mean and out[:, 1, ...] is the log variance
# target: Tensor of shape [B, 1, H, W]

# out = ...  # Your network output
# target = ...  # Your target values

# loss_fn = LogNormalNLLLoss()
# loss = loss_fn(out, target)