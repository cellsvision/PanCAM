import torch
from torch.nn.modules.loss import _Loss
from typing import List, Optional
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
import torch.nn as nn

__all__ = ["CMSELoss"]

class CMSELoss(_Loss):
    def __init__(self, mode: str, ignore_index: Optional[int] = None, reduction: str = 'mean'):

        super().__init__()
        # assert mode in {BINARY_MODE}
        self.ignore_index = ignore_index
        self.loss = nn.MSELoss(reduction=reduction).cuda()
        self.mode = mode  
    

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        # raise NotImplementedError("MSE loss is not implemented yet")

        if self.mode in {BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE}:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.loss(y_pred, y_true)

        # elif self.mode == MULTICLASS_MODE:
        #     raise NotImplementedError("Multiclass MSE loss is not implemented yet")

        #     num_classes = y_pred.size(1)
        #     loss = 0

        #     # Filter anchors with -1 label from loss computation
        #     if self.ignore_index is not None:
        #         not_ignored = y_true != self.ignore_index

        #     for cls in range(num_classes):
        #         cls_y_true = (y_true == cls).long()
        #         cls_y_pred = y_pred[:, cls, ...]

        #         if self.ignore_index is not None:
        #             cls_y_true = cls_y_true[not_ignored]
        #             cls_y_pred = cls_y_pred[not_ignored]

        #         loss += self.loss(cls_y_pred, cls_y_true)

        return loss
