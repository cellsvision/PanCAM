import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import PIL
import numpy as np
import pickle
import yaml
from albumentations.core.composition import Compose
from albumentations.augmentations import transforms, geometric
from albumentations.pytorch.transforms import ToTensorV2
import segmentation_models_pytorch as smp
from img_utils import save_mask, load_mask

import pandas as pd

color_palette = [0, 0, 0, 0, 64, 128, 64, 128, 0, 243, 152, 0]
color_palette = [0, 0, 0, 
                 0, 64, 128, 
                 64, 128, 0,  
                 243, 152, 0, 
                 128,128,255]

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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


def iou_score_per_class(output, target, multiclass=False):
    smooth = 1e-5

    n_class = target.shape[2]


    tmp_mask = np.zeros(target.shape, dtype=bool)
    if multiclass:
        for i in range(n_class):
            tmp_mask[:,:, i] = (output == (i+1))

        output_ = tmp_mask

    else:
        output_ = output > 0.5
    target_ = target > 0.5


    inter = (output_ & target_)
    un = (output_ | target_)


    inter_per_cls = []
    union_per_cls = []

    for i in range(inter.shape[2]):
        inter_per_cls.append(inter[:,:,i].sum())
        union_per_cls.append(un[:,:,i].sum())
    inter_per_cls= np.array(inter_per_cls)
    union_per_cls = np.array(union_per_cls)


    per_cls_result = (inter_per_cls + smooth) / (union_per_cls + smooth)

    per_cls_result[(inter_per_cls==0)&(union_per_cls==0)] = -1.


    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth), per_cls_result, inter_per_cls, union_per_cls


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        content = pickle.load(f)
    return content


def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        content = yaml.load(f)
    return content


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

def get_model(input_model_path, input_model_config_path):
    config = load_yaml(input_model_config_path)

    input_width = config['input_w']
    input_height = config['input_h']
    class_name_list = config['class_name']

    model = ARCH_DICT[config['arch']](
        encoder_name=config['encoder'],
        encoder_weights=None,
        classes=config['num_classes'],
        activation=config['act'],
    )

    model = nn.DataParallel(model)
    model.to('cuda:0')

    model.load_state_dict(torch.load(input_model_path))

    model.eval()


    return model, input_width, input_height, class_name_list


def get_crop_area(width, height, crop_size, overlap):
    current_crop_width = min(width, crop_size)
    current_crop_height = min(height, crop_size)
    assert crop_size > overlap, 'crop size must great than overlap: {} : {}'.format(str(crop_size),str(overlap))
    regions = []

    h_start = 0
    while h_start < height:
        w_start = 0
        while w_start < width:
            region_x2 = min(max(0, w_start + current_crop_width), width)
            region_y2 = min(max(0, h_start + current_crop_height), height)
            region_x1 = min(max(0, region_x2 - current_crop_width), width)
            region_y1 = min(max(0, region_y2 - current_crop_height), height)

            regions.append([region_x1, region_y1, region_x2, region_y2])

            # break when region reach the end
            if w_start + current_crop_width >= width: break

            w_start += current_crop_width - overlap

        # break when region reach the end
        if h_start + current_crop_height >= height: break

        h_start += current_crop_height - overlap

    regions = np.array(regions, dtype=int)
    return regions

def run_flip_rot_infer(img, preprocess, flip_op_list, model, cls_num, sigmoid=False):

    final_img = np.array(img)
    final_count_mask = np.zeros(final_img.shape[:2], dtype=int)
    final_whole_mask = np.zeros(tuple([cls_num]+list(final_img.shape[:2])), dtype=float)


    for op_index in range(2):

        current_single_img = np.array(final_img)
        current_single_count_mask = np.array(final_count_mask)
        current_single_whole_mask = np.array(final_whole_mask)

        if op_index == 1:
            current_single_img = np.rot90(current_single_img)
            current_single_count_mask = np.rot90(current_single_count_mask)
            current_single_whole_mask = np.transpose(current_single_whole_mask, (1,2,0))
            current_single_whole_mask = np.rot90(current_single_whole_mask)
            current_single_whole_mask = np.transpose(current_single_whole_mask, (2,0,1))

        for ops in flip_op_list:
            current_img = np.array(current_single_img)
            current_count_mask = np.array(current_single_count_mask)

            for sop_fn in ops:
                current_img = sop_fn(current_img)
                current_count_mask = sop_fn(current_count_mask)


            tensor_img = preprocess(image=current_img)['image'].unsqueeze(0).cuda()

            current_tiled_mask= model(tensor_img)
            if sigmoid:
                current_tiled_mask = torch.sigmoid(current_tiled_mask)



            current_tiled_mask = F.upsample(current_tiled_mask, (current_img.shape[0], current_img.shape[1]), mode='bilinear')

            current_whole_mask = current_tiled_mask.cpu().detach().numpy()[0]

            current_count_mask = current_count_mask+1

            for sop_fn in ops:
                current_img = sop_fn(current_img)
                current_count_mask = sop_fn(current_count_mask)

                current_whole_mask = np.transpose(current_whole_mask, (1,2,0))
                current_whole_mask = sop_fn(current_whole_mask)
                current_whole_mask = np.transpose(current_whole_mask, (2,0,1))


            current_single_count_mask = current_count_mask
            current_single_whole_mask = current_single_whole_mask + current_whole_mask


        if op_index == 1:
            current_single_img = np.rot90(current_single_img, -1)
            current_single_count_mask = np.rot90(current_single_count_mask, -1)
            current_single_whole_mask = np.transpose(current_single_whole_mask, (1,2,0))
            current_single_whole_mask = np.rot90(current_single_whole_mask, -1)
            current_single_whole_mask = np.transpose(current_single_whole_mask, (2,0,1))


        final_count_mask = current_single_count_mask
        final_whole_mask = current_single_whole_mask


    return final_count_mask, final_whole_mask









def main(input_img_path, input_label_path, input_model_path, input_model_config_path, 
         output_mask_path, output_csv_path, 
         img_rgb_mean, img_rgb_std, crop_size_list, overlap_list,
        use_cache=False, sigmoid=False, target_wsi_file_list=None):

    model, input_width, input_height, class_name_list = get_model(input_model_path, input_model_config_path)

    cls_num = len(class_name_list)

    preprocess = Compose([
        geometric.resize.Resize(input_height, input_width),
        transforms.Normalize(mean=img_rgb_mean, std=img_rgb_std),
        ToTensorV2(),
    ])


    flip_op_list = [[],[np.flipud],[np.fliplr], [np.flipud, np.fliplr]]


    if target_wsi_file_list is None: 
        img_file_list = scan_files(input_img_path, ext_list = ['.png'])
    else:
        img_file_list = []
        for tmp_dir in target_wsi_file_list:
            tmp_file_list = scan_files(os.path.join(input_img_path, tmp_dir), ext_list = ['.png'], replace_root=input_img_path)
            img_file_list.extend(tmp_file_list)

    

    min_side = max(crop_size_list)

    iou_score_list = []
    iou_per_class_score_list = []
    per_cls_inter_list = []
    per_cls_union_list = []
    img_name_list = []

    for it, f_path in enumerate(img_file_list):
        print('process image {}/{}: {}'.format(str(it), str(len(img_file_list)), f_path))
        img_path = os.path.join(input_img_path, f_path)
        label_path = os.path.join(input_label_path, os.path.splitext(f_path)[0] + '.png')

        current_output_path = os.path.join(output_mask_path, os.path.splitext(f_path)[0]+'.png')

        if use_cache and os.path.isfile(current_output_path):
            index_mask = load_mask(current_output_path)

        else:
            whole_img = cv2.imread(img_path)[:,:,::-1]
            print(whole_img.shape)
            # whole_img = cv2.imread(img_path)
            img_h, img_w = whole_img.shape[:2]
            short_side = min(img_h, img_w)
            count_mask = np.zeros(whole_img.shape[:2], dtype=int)
            whole_mask = np.zeros(tuple([cls_num]+list(whole_img.shape[:2])), dtype=float)

            if short_side < min_side:

                current_single_count_mask,current_single_whole_mask = run_flip_rot_infer(whole_img, preprocess, flip_op_list, model, cls_num, sigmoid=sigmoid)

                count_mask = count_mask + current_single_count_mask
                whole_mask = whole_mask + current_single_whole_mask



            else:
                for tmp_crop_size, tmp_overlap in zip(crop_size_list, overlap_list):

                    regions = get_crop_area(whole_img.shape[1], whole_img.shape[0], tmp_crop_size, tmp_overlap)

                    for region in regions:
                        x1, y1, x2, y2 = region

                        tiled_img = np.array(whole_img[y1:y2,x1:x2])
                        current_single_count_mask,current_single_whole_mask = run_flip_rot_infer(tiled_img, preprocess, flip_op_list, model, cls_num, sigmoid=sigmoid)
                        count_mask[y1:y2,x1:x2] = count_mask[y1:y2,x1:x2] + current_single_count_mask
                        whole_mask[:,y1:y2,x1:x2] = whole_mask[:,y1:y2,x1:x2] + current_single_whole_mask
                        # print('### debug max check')


            final_whole_mask = whole_mask/count_mask

            print(final_whole_mask.shape)

    
            invalid_mask = (np.sum(final_whole_mask < 0.5, axis=0) == cls_num)

            print(np.unique(final_whole_mask, return_counts=True))
            print(np.max(final_whole_mask))

            if final_whole_mask.shape[0] == 1:
                index_mask = np.zeros(final_whole_mask.shape[1:])
                index_mask = index_mask.astype(np.uint8)
                index_mask[final_whole_mask[0]>=0.5] = 1
            else:
                index_mask = np.argmax(final_whole_mask, axis=0)
                index_mask = index_mask.astype(np.uint8)
                index_mask = index_mask+1
                index_mask[invalid_mask] = 0


            if not os.path.isdir(os.path.dirname(current_output_path)):
                os.makedirs(os.path.dirname(current_output_path))
            save_mask(current_output_path, index_mask, color_palette)
    



if __name__ == '__main__':
    input_model_path = 'path_to_model_dir'
    input_model_config_path = 'path_to_model_dir/config.yml'
    input_target_wsi_csv = 'path_to_target_wsi.csv'
    model_name_list = ['xx1.pth', 'xx2.pth', 'xx3.pth']
    input_img_path = 'path_to_input_images'
    input_label_path = 'path_to_input_masks'
    output_mask_path = 'path_to_output_masks'
    csv_output = 'path_to_output_csv'

    if input_target_wsi_csv is not None:
        test_data = pd.read_csv(input_target_wsi_csv)
        target_wsi_file_list = list(test_data.to_numpy()[:, 0])
    else:
        target_wsi_file_list=None

    add_sigmoid = True

    # imagenet
    img_rgb_mean = [0.485, 0.456, 0.406]
    img_rgb_std = [0.229, 0.224, 0.225]

    base_crop_size = 2048
    scale_list = [1]

    crop_size = [int(base_crop_size * current_scale) for current_scale in scale_list]
    overlap = [0]

    print(crop_size)
    print(overlap)

    for current_mode in model_name_list:
        current_input_model_path = os.path.join(input_model_path, current_mode)
        current_output_mask_path = os.path.join(output_mask_path, os.path.splitext(current_mode)[0])
        current_output_csv_path = os.path.join(csv_output, os.path.splitext(current_mode)[0]+'.csv')
        main(input_img_path, input_label_path, current_input_model_path, input_model_config_path, 
             current_output_mask_path, current_output_csv_path, 
             img_rgb_mean, img_rgb_std, crop_size, overlap, sigmoid=add_sigmoid, target_wsi_file_list=target_wsi_file_list)
