import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import *

ARCH_DICT = {'unetplusplus': smp.UnetPlusPlus, 
             'unet': smp.Unet, 
             'fpn': smp.FPN, 
             'pspnet': smp.PSPNet,
             'pan': smp.PAN,
             'manet': smp.MAnet,
             'linknet': smp.Linknet,
             'deeplabv3': smp.DeepLabV3, 
             'deeplabv3plus': smp.DeepLabV3Plus
            }

LOSS_DICT = { 'jaccard': JaccardLoss,
              'dice': DiceLoss,
              'focal': FocalLoss,
              'lovasz': LovaszLoss,
              'soft_bce': SoftBCEWithLogitsLoss,
              'soft_ce': SoftCrossEntropyLoss,
              'tversky': TverskyLoss,
              'focal_tversky': FocalTverskyLoss
            }