from torch import nn
from torch.nn.modules.loss import _Loss

__all__ = ["JointLoss", "WeightedLoss", 'JointLoss_v2']


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, loss_list, loss_list_weight):
        super().__init__()
        self.loss_list = [WeightedLoss(l, lw) for l,lw in zip(loss_list, loss_list_weight)]


    def forward(self, *input):
        total_loss = 0.
        for l in self.loss_list:
            total_loss += l(*input)
        

        return total_loss

class JointLoss_v2(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, loss_list, loss_list_weight, loss_name_list):
        super().__init__()
        self.loss_list = [WeightedLoss(l, lw) for l,lw in zip(loss_list, loss_list_weight)]
        self.loss_name_list = loss_name_list

    def forward(self, *input):
        total_loss = 0.
        
        pred_result = input[0]
        gt_result = input[1]

        loss_dict = {}
        
        for l, ln in zip(self.loss_list, self.loss_name_list):
            if ln == 'cmse':
                tmp_loss = l(pred_result, gt_result['mse'])

            else:
                tmp_loss = l(pred_result, gt_result['general'])
            
            total_loss = total_loss + tmp_loss
            loss_dict[ln] = tmp_loss
        
            


        return total_loss, loss_dict