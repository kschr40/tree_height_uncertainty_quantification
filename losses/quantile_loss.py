import torch.nn as nn
import torch


class QuantileLoss(nn.Module):
    """Mean Absolute error"""

    def __init__(
        self,
        ignore_value=None,
        pre_calculation_function=None,
        quantiles=[0.5,0.1,0.9],
    ):
        super().__init__()
        self.ignore_value = ignore_value
        self.pre_calculation_function = pre_calculation_function
        self.quantiles = quantiles
    def forward(self, out, target):
        """
        Applies the L1 loss
        :param out: output of the network
        :param target: target
        :return: l1 loss
        """
        if self.pre_calculation_function != None:
            out, target = self.pre_calculation_function(out, target)
        quant_dict = []
        for i in range(len(self.quantiles)):
            quant_dict.append(out[:,i,...].flatten())

        target = target.flatten()

        if self.ignore_value is not None:
            mask = target != self.ignore_value
            for i in range(len(quant_dict)):
                quant_dict[i] = quant_dict[i][mask]
            target = target[mask]
        loss_list = []
        loss_tensor = torch.zeros(len(quant_dict), requires_grad=True)
        loss = torch.tensor(0.0, requires_grad=True)
        for i, quant in enumerate(quant_dict):
            current_loss = torch.max((quant-1) * (target - quant), quant * (target - quant))
            # loss_tensor[i] += current_loss.mean()
            loss_list.append(current_loss.mean())
            # loss += current_loss.mean()
        # return (loss_quant0 + loss_quant1 + loss_quant2 + quant_error0 + quant_error1).mean()
        return torch.mean(torch.stack(loss_list))


    # def forward(self, out, target):
    #     """
    #     Applies the L1 loss
    #     :param out: output of the network
    #     :param target: target
    #     :return: l1 loss
    #     """
    #     if self.pre_calculation_function != None:
    #         out, target = self.pre_calculation_function(out, target)

    #     quant0_pred = out[:,0,...].flatten()
    #     quant1_pred = out[:,1,...].flatten()
    #     quant2_pred = out[:,2,...].flatten()
    #     target = target.flatten()

    #     if self.ignore_value is not None:
    #         quant0_pred = quant0_pred[target != self.ignore_value]
    #         quant1_pred = quant1_pred[target != self.ignore_value]
    #         quant2_pred = quant2_pred[target != self.ignore_value]
    #         target = target[target != self.ignore_value]

    #     quant0 = self.quantiles[0]
    #     quant1 = self.quantiles[1]
    #     quant2 = self.quantiles[2]

    #     loss_quant0 = torch.max((quant0-1) * (target - quant0_pred), quant0 * (target - quant0_pred))
    #     loss_quant1 = torch.max((quant1-1) * (target - quant1_pred), quant1 * (target - quant1_pred))
    #     loss_quant2 = torch.max((quant2-1) * (target - quant2_pred), quant2 * (target - quant2_pred))

    #     quant_error0 = torch.relu(quant0_pred - quant2_pred)  # Penalize if the 0.9 quantile is below the 0.5 quantile
    #     quant_error1 = torch.relu(quant1_pred - quant0_pred)  # Penalize if the 0.1 quantile is above the 0.5 quantile
    #     if (loss_quant0 + loss_quant1 + loss_quant2 + quant_error0 + quant_error1).mean() < 0:
    #         print("Negative quantile loss, something is wrong!")
    #     # return (loss_quant0 + loss_quant1 + loss_quant2 + quant_error0 + quant_error1).mean()
    #     return (loss_quant0 + loss_quant1 + loss_quant2).mean()
