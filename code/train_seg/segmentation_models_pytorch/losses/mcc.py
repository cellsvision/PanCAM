import torch
from torch.nn.modules.loss import _Loss
from typing import List, Optional
from .constants import BINARY_MODE

class MCCLoss(_Loss):
    def __init__(self, mode: str, eps: float = 1e-5, ignore_index: Optional[int] = None):
        """Compute Matthews Correlation Coefficient Loss for image segmentation task.
        It only supports binary mode.

        Args:
            eps (float): Small epsilon to handle situations where all the samples in the dataset belong to one class

        Reference:
            https://github.com/kakumarabhishek/MCC-Loss
        """
        super().__init__()
        assert mode in {BINARY_MODE}
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute MCC loss

        Args:
            y_pred (torch.Tensor): model prediction of shape (N, H, W) or (N, 1, H, W)
            y_true (torch.Tensor): ground truth labels of shape (N, H, W) or (N, 1, H, W)

        Returns:
            torch.Tensor: loss value (1 - mcc)
        """

        bs = y_true.shape[0]

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)


        if self.ignore_index is not None:
            pad_mask = y_true.eq(self.ignore_index)
            # y_true = y_true.masked_fill(pad_mask, 0)

            tp = torch.sum(torch.mul(y_pred.masked_fill(pad_mask, 0), y_true.masked_fill(pad_mask, 0))) + self.eps
            tn = torch.sum(torch.mul((1 - y_pred).masked_fill(pad_mask, 0), (1 - y_true).masked_fill(pad_mask, 0))) + self.eps
            fp = torch.sum(torch.mul(y_pred.masked_fill(pad_mask, 0), (1 - y_true).masked_fill(pad_mask, 0))) + self.eps
            fn = torch.sum(torch.mul((1 - y_pred).masked_fill(pad_mask, 0), y_true.masked_fill(pad_mask, 0))) + self.eps
        else:
            tp = torch.sum(torch.mul(y_pred, y_true)) + self.eps
            tn = torch.sum(torch.mul((1 - y_pred), (1 - y_true))) + self.eps
            fp = torch.sum(torch.mul(y_pred, (1 - y_true))) + self.eps
            fn = torch.sum(torch.mul((1 - y_pred), y_true)) + self.eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(torch.add(tp, fp) * torch.add(tp, fn) * torch.add(tn, fp) * torch.add(tn, fn))

        mcc = torch.div(numerator.sum(), denominator.sum())
        loss = 1.0 - mcc

        return loss
