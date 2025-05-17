import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def iou_score_per_class(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    inter = (output_ & target_)
    un = (output_ | target_)
    
    inter_per_cls = []
    union_per_cls = []

    for i in range(inter.shape[1]):
        inter_per_cls.append(inter[:,i].sum())
        union_per_cls.append(un[:,i].sum())
    inter_per_cls= np.array(inter_per_cls)
    union_per_cls = np.array(union_per_cls)

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth), (inter_per_cls + smooth) / (union_per_cls + smooth)

def iou_score_per_class_multiclass(output, target, ignore_label=7):
    smooth = 1e-5


    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()


    # print('### debug output result:')
    # print(output.shape)

    # print('### debug target result:')
    # print(target.shape)
    # print(np.unique(target, return_counts=True))

    tmp_target = np.zeros(output.shape)
    for i in range(tmp_target.shape[1]):
        tmp_target[:,i,:,:][target==i] = 1
        output[:,i,:,:][target==ignore_label] = 0


    output_ = output > 0.5
    target_ = tmp_target > 0.5
    inter = (output_ & target_)
    un = (output_ | target_)
    
    inter_per_cls = []
    union_per_cls = []

    for i in range(inter.shape[1]):
        inter_per_cls.append(inter[:,i].sum())
        union_per_cls.append(un[:,i].sum())
    inter_per_cls= np.array(inter_per_cls)
    union_per_cls = np.array(union_per_cls)

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth), (inter_per_cls + smooth) / (union_per_cls + smooth)

def iou_score_per_class_multiclass_v2(output, target, ignore_label=5):
    smooth = 1e-5


    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()



    result = np.argmax(output, axis=1)

    result[target==ignore_label] = ignore_label
    
    tmp_target = np.zeros(output.shape, dtype=bool)
    tmp_output = np.zeros(output.shape, dtype=bool)


    for i in range(tmp_target.shape[1]):
        tmp_target[:,i,:,:][target==i] = 1
        tmp_output[:,i,:,:][result==i] = 1

    inter = (tmp_target & tmp_output)
    un = (tmp_target | tmp_output)
    
    inter_per_cls = []
    union_per_cls = []

    for i in range(inter.shape[1]):
        inter_per_cls.append(inter[:,i].sum())
        union_per_cls.append(un[:,i].sum())
    inter_per_cls= np.array(inter_per_cls)
    union_per_cls = np.array(union_per_cls)

    intersection = (tmp_output & tmp_target).sum()
    union = (tmp_output | tmp_target).sum()

    no_bg_inter = (tmp_output[:,1:] & tmp_target[:, 1:]).sum()
    no_bg_union = (tmp_output[:,1:] | tmp_target[:, 1:]).sum()

    return (intersection + smooth) / (union + smooth), (no_bg_inter + smooth) / (no_bg_union + smooth),(inter_per_cls + smooth) / (union_per_cls + smooth)

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
