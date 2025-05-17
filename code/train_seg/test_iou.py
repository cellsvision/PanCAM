from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch.nn.parallel
import argparse
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
import pandas as pd
Image.MAX_IMAGE_PIXELS = None


def scan_files(input_file_path, ext_list = ['.txt'], replace_root=True):
    file_list = []
    for root, dirs, files in os.walk(input_file_path):
        # scan all files and put it into list

        for f in files:
            if os.path.splitext(f)[1].lower() in ext_list:
                if replace_root is True:
                    file_list.append(os.path.join(root, f).replace("\\","/").replace(os.path.join(input_file_path, "").replace("\\","/"), "", 1 ))
                elif replace_root is not False:
                    file_list.append(os.path.join(root, f).replace("\\","/").replace(os.path.join(replace_root, "").replace("\\","/"), "", 1 ))
                else:
                    file_list.append(os.path.join(root, f).replace("\\","/"))

    return file_list

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_confusion_matrix(seg_gt, pred, filename, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    # pred = pred.numpy()
    seg_pred = np.asarray(pred, dtype=np.uint8)
    seg_pred_h, seg_pred_w = seg_pred.shape[0], seg_pred.shape[1]
    seg_pred = seg_pred.reshape(-1, 1)

    # seg_gt = label.numpy()
    seg_gt_h, seg_gt_w = seg_gt.shape[0], seg_gt.shape[1]
    seg_gt = seg_gt.reshape(-1, 1)

    if seg_pred_h != seg_gt_h or seg_pred_w != seg_gt_w:
        with open('test_iou_scale_not_match_0125.txt', 'a') as f:
            print('scale not match:', filename, 'seg_pred_h:', seg_pred_h, 'seg_pred_w:', seg_pred_w, 'seg_gt_h:', seg_gt_h, 'seg_gt_w:', seg_gt_w, file=f)

    cn_gt = np.where(seg_gt == 1)[0]
    cn_gt_number = len(cn_gt)

    cn_pred = np.where(seg_pred == 1)[0]
    cn_pred_number = len(cn_pred)

    tp = list(set(cn_gt).intersection(set(cn_pred)))
    tp_number = len(tp)
    if cn_gt_number == 0 and cn_pred_number == 0:
        iou_per = 1
    else:
        iou_per = tp_number / float(cn_gt_number + cn_pred_number - tp_number)
    print('iou_per: %.3f' % iou_per)
    print('tp_per: %d' % tp_number)
    print('cn_gt_number_per: %d' % cn_gt_number)
    print('cn_pred_number_per: %d' % cn_pred_number)
    max_length = max(seg_pred_h, seg_pred_w)
    min_length = min(seg_pred_h, seg_pred_w)
    print('max_length: %d' % max_length)
    print('min_length: %d' % min_length)

    '''
    fp = list(set(cn_pred).difference(set(cn_gt)))
    fp_number = len(fp)
    fn = list(set(cn_gt).difference(set(cn_pred)))
    fn_number = len(fn)
    dice_per = (2 * tp_number) / float(2*tp_number + fp_number + fn_number)
    print('dice_per: %.3f' % dice_per)
    '''

    #seg_gt = np.expand_dims(seg_gt, 0)
    #ignore_index = seg_gt != ignore
    #seg_gt = seg_gt[ignore_index]
    #seg_pred = seg_pred[ignore_index]

    #cm = confusion_matrix(seg_gt, seg_pred, labels=[0, 1])

    '''
    index = (seg_gt * num_class + seg_pred).astype('int32')
    index = np.squeeze(index, 0)
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    '''
    return iou_per, 0, tp_number, cn_gt_number, cn_pred_number, max_length, min_length

# test_label = './get_bladder_train_val_split/bladder_test_label.pkl'

test_label_path = '/public/lz_dataset/bladder/bladder_hist/dataset_bladder_20220122/png_mask_no_border'
test_target_csv = 'bladder_val_wsi.csv'

# pred_label_path = '/public/lz_dataset/bladder/bladder_hist/dataset_bladder_20220122_result_20220207/model'
pred_label_path = '/public/lz_dataset/bladder/bladder_hist/dataset_bladder_20220122_result_20220208_exp_1/model'
pred_label_path = '/public/lz_dataset/bladder/bladder_hist/dataset_bladder_20220122_result_20220217_exp_1/model'
pred_label_path = '/public/lz_dataset/bladder/bladder_hist/dataset_bladder_20220122_result_20220221_exp_1/best_val_iou'
pred_label_path = '/public/lz_dataset/bladder/bladder_hist/dataset_bladder_20220122_result_20220221_exp_1/best_test_iou'
pred_label_path = '/public/lz_dataset/bladder/bladder_hist/dataset_bladder_20220122_result_20220221_exp_1/199'
output_path = './bladder_dataset_0128_20220221_exp_1_199_result'
# output_path = './bladder_dataset_0128_20220217_exp_1_result'

test_data = pd.read_csv(test_target_csv)
target_wsi_file_list = list(test_data.to_numpy()[:, 0])

# confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
iou_all = []
dice_all = []
tp_all = []
cn_gt_all = []
cn_pred_all = []
test_name_all = []
max_all = []
min_all = []

img_file_list = []
for tmp_dir in target_wsi_file_list:
    tmp_file_list = scan_files(os.path.join(test_label_path, tmp_dir), ext_list = ['.png'], replace_root=test_label_path)
    img_file_list.extend(tmp_file_list)

for index, target_file in enumerate(img_file_list):
    print('{}/{}: {}'.format(str(index), len(img_file_list), target_file))
    whole_mask = Image.open(os.path.join(test_label_path, target_file))
    whole_mask = np.array(whole_mask).astype(np.uint8)
    print('### debug test')
    print(np.unique(whole_mask, return_counts=True))
    
    pred = Image.open(os.path.join(pred_label_path, target_file))
    pred = np.array(pred).astype(np.uint8)
    print('### debug pred')
    print(np.unique(pred, return_counts=True))

    test_name_all.append(os.path.basename(target_file))
        #print(img.size)
    #except Exception as e:
        #with open('wrong.txt', 'a') as f:
            #print(label_path, file=f)
    

    # pred = torch.from_numpy(pred_np)
    #pred = pred.unsqueeze(0)

    if not isinstance(pred, (list, tuple)):
        pred = [pred]
    for i, x in enumerate(pred):
        iou_per, dice_per, tp_number, cn_gt_number, cn_pred_number, max_len, min_len = get_confusion_matrix(whole_mask, x, os.path.basename(target_file))
        iou_all.append(iou_per)
        dice_all.append(dice_per)
        tp_all.append(tp_number)
        cn_gt_all.append(cn_gt_number)
        cn_pred_all.append(cn_pred_number)
        max_all.append(max_len)
        min_all.append(min_len)

mean_iou = sum(iou_all) / len(iou_all)
mean_dice = sum(dice_all) / len(dice_all)
print('mean_iou:', mean_iou)
print('mean_dicr:', mean_dice)
tp_total = sum(tp_all)
cn_gt_total = sum(cn_gt_all)
cn_pred_total = sum(cn_pred_all)
iou_average = tp_total / float(cn_gt_total + cn_pred_total - tp_total)
print('iou_average:', iou_average)
print('iou_number:', len(iou_all))

save_dict = {'test_files': test_name_all, 'iou_all': iou_all, 'dice_all': dice_all, 'tp_all': tp_all, 'cn_gt_all': cn_gt_all, 'cn_pred_all': cn_pred_all, 'max_all': max_all, 'min_all': min_all, 'mean_iou': mean_iou, 'mean_dice': mean_dice, 'iou_average': iou_average}

if not os.path.isdir(output_path):
    os.makedirs(output_path)
output_dict = open(os.path.join(output_path, 'result.pkl'), 'wb')
pickle.dump(save_dict, output_dict)

result = pd.DataFrame()
result['test_files'] = test_name_all
result['iou_value'] = iou_all
# result['dice_value'] = dice_all
result['tp_number'] = tp_all
result['cn_gt_number'] = cn_gt_all
result['cn_pred_number'] = cn_pred_all
result['max_length'] = max_all
result['min_length'] = min_all
result.to_csv(os.path.join(output_path, 'result.csv'), index=None)

'''
for i in range(nums):
    pos = confusion_matrix[..., i].sum(1)
    res = confusion_matrix[..., i].sum(0)
    tp = np.diag(confusion_matrix[..., i])
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print(IoU_array)
    print(mean_IoU)
'''


