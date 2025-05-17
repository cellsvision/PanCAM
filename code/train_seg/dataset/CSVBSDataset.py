import os
import random

import cv2
import numpy as np
import torch
import pickle
import math

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        content = pickle.load(f)
    return content


def load_png(file_path):
    pil_obj = Image.open(file_path)
    return np.array(pil_obj)


def scan_files(input_file_path, ext_list = ['.txt'], replace_root=True):
    file_list = []
    for root, dirs, files in os.walk(input_file_path):
        # scan all files and put it into list

        for f in files:
            if os.path.splitext(f)[1].lower() in ext_list:
                if replace_root:
                    file_list.append(os.path.join(root, f).replace("\\","/").replace(os.path.join(input_file_path, "").replace("\\","/"), "", 1 ))
                else:
                    file_list.append(os.path.join(root, f).replace("\\","/"))

    return file_list

def save_mask(out_mask_path, current_mask, palette_list= [ 0, 0, 0, 
                                                           255, 128, 0,
                                                           128,128,128,]):
    if len(current_mask.shape) == 3:
        pil_mat = np.squeeze(current_mask, axis=2)
    else:
        pil_mat = current_mask
    pil_img = Image.fromarray(pil_mat)
    pil_img = pil_img.convert("P")
    pil_img.putpalette(palette_list)

    if not os.path.isdir(os.path.dirname(out_mask_path)): 
        os.makedirs(os.path.dirname(out_mask_path))

    pil_img.save(out_mask_path)

# find pure bg img in labeled data
def find_pure_bg_img(mask_dir, img_ids, mask_ext='.png', auto_bg=False):
    bg_check_list = []


    for img_id in img_ids:
       
        mask_path = os.path.join(mask_dir, os.path.splitext(img_id)[0]+mask_ext)
        
        if os.path.isfile(mask_path):
            cmask = load_png(mask_path)
            sum_mask = np.sum(cmask)
            bg_check = 1 if sum_mask == 0 else 0
        elif auto_bg:
            bg_check = 1
        else:
            raise Exception('no mask found for image {}, either check the datasets or set auto_bg to True'.format(img_id))


        bg_check_list.append(bg_check)
    
    return np.array(bg_check_list)


def get_regions(img_shape, crop_size, overlap, area_map_params=None):
    regions = []

    assert img_shape[1] >= crop_size and img_shape[0] >= crop_size
    assert crop_size > overlap
    h_start = 0


    height = img_shape[0]
    width = img_shape[1]

    if area_map_params is not None:
        valid_area_map = area_map_params
        area_patch = height/valid_area_map.shape[0]


    while h_start < img_shape[0]:
        w_start = 0
        while w_start < img_shape[1]:
            region_x2 = min(max(0, w_start + crop_size), img_shape[1])
            region_y2 = min(max(0, h_start + crop_size), img_shape[0])
            region_x1 = min(max(0, region_x2 - crop_size), img_shape[1])
            region_y1 = min(max(0, region_y2 - crop_size), img_shape[0])

            if area_map_params is not None:
                area_map_x1 = math.floor(region_x1 / area_patch)
                area_map_y1 = math.floor(region_y1 / area_patch)
                area_map_x2 = math.ceil(region_x2 / area_patch)
                area_map_y2 = math.ceil(region_y2 / area_patch)


                if np.max(valid_area_map[area_map_y1:area_map_y2, area_map_x1:area_map_x2]) > 0:
                    regions.append([region_x1, region_y1, region_x2, region_y2])
            else:
                regions.append([region_x1, region_y1, region_x2, region_y2])

            # break when region reach the end
            if w_start + crop_size >= img_shape[1]: break

            w_start += crop_size - overlap

        # break when region reach the end
        if h_start + crop_size >= img_shape[0]: break

        h_start += crop_size - overlap

    # regions = np.array(regions, dtype=np.float32)
    return regions



"""
CSVBSDatasetV1: Basic version
- Implements basic random crop data sampling strategy
- Supports multi-scale training with random_image_size
- Handles three types of data:
  1. Labeled pure background (pure_bg_ratio)
  2. Mixed labeled data (mix_ratio)
  3. Labeled foreground (pos_ratio = 1 - pure_bg_ratio - mix_ratio)
- Basic data augmentation support
"""

class CSVBSDatasetV1(torch.utils.data.Dataset):
    def __init__(self, data_root_path, 
                 wsi_data_list_path, neg_wsi_data_list_path,
                 wsi_data_details_info_dir=None,
                 img_ext='.png', mask_ext='.png', target_patch_num=5000,
                 target_image_size=2048, 
                 random_image_size=[1024, 3072], random_size_ratio=0.2,
                 no_label_neg_ratio=0.3, no_label_neg_data_root_path=None,
                 pure_bg_ratio=0.15, mix_ratio=0.15, ignore_value=2,  
                 transform=None, preprocessing=None, cust_trans=False,
                 try_limit=100):

        self.img_dir = os.path.join(data_root_path, 'img')
        self.mask_dir = os.path.join(data_root_path, 'mask')
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.try_limit = try_limit

        self.target_path_num = target_patch_num
        self.target_img_size = target_image_size
        self.pure_bg_ratio = pure_bg_ratio

        # image process
        self.transform = transform
        self.preprocessing = preprocessing
        self.cust_trans = cust_trans

        # random crop setting
        self.random_size_ratio = random_size_ratio
        self.random_image_size = random_image_size

        with open(wsi_data_list_path, 'r') as f:
            self.wsi_data_list = f.read().splitlines() 

        self.wsi_data_list = [f.rstrip() for f in self.wsi_data_list]


        # process fg files
        self.fg_wsi_file_list = []
        self.fg_wsi_file_details_dict = {}
        fg_detials_path = os.path.join(wsi_data_details_info_dir, 'fg')
        fg_details_info_file_list = scan_files(fg_detials_path, replace_root=False)
        for tmp_f_path in fg_details_info_file_list:
            tmp_wsi_name = os.path.splitext(os.path.basename(tmp_f_path))[0]            
            
            if tmp_wsi_name in self.wsi_data_list:        
                with open(tmp_f_path, 'r') as f:
                    tmp_details_file_list = f.read().splitlines()         
                self.fg_wsi_file_list.append(tmp_wsi_name)
                self.fg_wsi_file_details_dict[tmp_wsi_name] = tmp_details_file_list


        # process bg files
        self.bg_wsi_file_list = []
        self.bg_wsi_file_details_dict = {}
        bg_detials_path = os.path.join(wsi_data_details_info_dir, 'no_fg')
        bg_details_info_file_list = scan_files(bg_detials_path, replace_root=False)
        for tmp_f_path in bg_details_info_file_list:
            tmp_wsi_name = os.path.splitext(os.path.basename(tmp_f_path))[0]            
            
            if tmp_wsi_name in self.wsi_data_list:        
                with open(tmp_f_path, 'r') as f:
                    tmp_details_file_list = f.read().splitlines()         
                self.bg_wsi_file_list.append(tmp_wsi_name)
                self.bg_wsi_file_details_dict[tmp_wsi_name] = tmp_details_file_list


        # merge all details
        wsi_file_details_dict = {}
        for tmp_wsi_f in self.wsi_data_list:     
            tmp_file_details_list = []       
            if tmp_wsi_f in self.fg_wsi_file_details_dict:
                tmp_file_details_list.extend(self.fg_wsi_file_details_dict[tmp_wsi_f])

            if tmp_wsi_f in self.bg_wsi_file_details_dict:
                tmp_file_details_list.extend(self.bg_wsi_file_details_dict[tmp_wsi_f])
    
            if len(tmp_file_details_list) > 0:
                wsi_file_details_dict[tmp_wsi_f] = tmp_file_details_list

        self.wsi_file_details_dict = wsi_file_details_dict

        # process unlabeled pure neg data
        with open(neg_wsi_data_list_path, 'r') as f:
            self.neg_wsi_data_list = f.read().splitlines() 

        self.neg_img_dir = os.path.join(no_label_neg_data_root_path, 'img')
        self.neg_mask_dir = os.path.join(no_label_neg_data_root_path, 'mask')
        self.neg_ratio = no_label_neg_ratio

        self.neg_loc_name_dict = {}
        final_valid_neg_wsi_data_list = []
        for tmp_wsi in self.neg_wsi_data_list:
            tmp_img_path = os.path.join(self.neg_img_dir, tmp_wsi)
            tmp_details_img_list = scan_files(tmp_img_path, ext_list=['.png'], replace_root=True)
            if len(tmp_details_img_list) > 0 :
                self.neg_loc_name_dict[tmp_wsi] = tmp_details_img_list
                final_valid_neg_wsi_data_list.append(tmp_wsi)
        self.neg_wsi_data_list = final_valid_neg_wsi_data_list

        self.mix_ratio = mix_ratio
        self.ignore_value = ignore_value
        self.pos_ratio = 1.-(self.neg_ratio+self.pure_bg_ratio+self.mix_ratio)
      

            
    def __len__(self):
        return self.target_path_num

    def __getitem__(self, idx):

        current_type = random.choices([0,1,2,3], 
                       weights=[self.neg_ratio, 
                                self.pure_bg_ratio,
                                self.mix_ratio,
                                self.pos_ratio])[0]
        # 0: no label neg, which from cropped bg image, no random size
        # 1: labeled pure bg
        # 2: labeled data include bg and fg
        # 3: labeled fg 

        if current_type == 0:
            target_wsi = random.choice(self.neg_wsi_data_list)
            tmp_loc = random.choice(self.neg_loc_name_dict[target_wsi])
            img_path = os.path.join(self.neg_img_dir, target_wsi, tmp_loc)
            mask_path = os.path.join(self.neg_mask_dir, target_wsi, tmp_loc)

            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = np.array(Image.open(mask_path))

        else:  
            break_loop = True
            try_count = 0
            
            while True:
                if current_type == 1:
                    target_wsi = random.choice(self.bg_wsi_file_list)
                    tmp_loc = random.choice(self.bg_wsi_file_details_dict[target_wsi])
                    img_path = os.path.join(self.img_dir, target_wsi, tmp_loc)
                    mask_path = os.path.join(self.mask_dir, target_wsi, tmp_loc)            
                elif current_type == 2:
                    target_wsi = random.choice(self.wsi_data_list)
                    tmp_loc = random.choice(self.wsi_file_details_dict[target_wsi])
                    img_path = os.path.join(self.img_dir, target_wsi, tmp_loc)
                    mask_path = os.path.join(self.mask_dir, target_wsi, tmp_loc)                
                else:
                    target_wsi = random.choice(self.fg_wsi_file_list)
                    tmp_loc = random.choice(self.fg_wsi_file_details_dict[target_wsi])
                    img_path = os.path.join(self.img_dir, target_wsi, tmp_loc)
                    mask_path = os.path.join(self.mask_dir, target_wsi, tmp_loc)        

                whole_img_array = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                whole_mask_array = load_png(mask_path)

                h,w = whole_img_array.shape[:2]

                current_target_size = min(h,w)



                # process random crop
                if random.random() < self.random_size_ratio:
                    crop_size = random.randint(min(self.random_image_size[0], current_target_size), 
                                            min(self.random_image_size[1], current_target_size))
                else:
                    crop_size = self.target_img_size

                y_range = h - crop_size
                x_range = w - crop_size

                start_x = random.randint(0, max(0,x_range-1))
                start_y = random.randint(0, max(0,y_range-1))
                end_x = start_x + crop_size
                end_y = start_y + crop_size

                patch_mask_array = np.array(whole_mask_array[start_y:end_y, start_x:end_x])
                
                img = np.array(whole_img_array[start_y:end_y, start_x:end_x])

                mask = patch_mask_array


                check_img = np.array(img)
                r_check_mask = check_img[:,:,0] > 220
                g_check_mask = check_img[:,:,1] > 220
                b_check_mask = check_img[:,:,2] > 220
                white_ignore_mask = r_check_mask&g_check_mask&b_check_mask

                mask = np.array(mask)
                fg_mask = mask == 1
                mask[(white_ignore_mask & fg_mask)] = self.ignore_value

                if current_type == 3:
                    num_pos = np.sum(mask==1)
                    pos_ratio = num_pos/(crop_size*crop_size)
                    # print('### debug type 3', current_type)
                    # print(num_pos, pos_ratio)
                    if pos_ratio > 0.05:
                        break_loop = True
                    else:
                        break_loop = False
                else:
                    break_loop = True

                try_count += 1

                if break_loop or try_count > self.try_limit:
                    break

        # generate random crop done
        ori_img = np.array(img)


        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
        else:
            img = img.astype('float32') / 255
            img = img.transpose(2, 0, 1)


        mask = mask.astype('float32')
        
        # print('### debug mask')
        # print(np.unique(mask, return_counts=True))

        ori_img = cv2.resize(ori_img, (img.shape[2], img.shape[1]) )
        ori_img = ori_img.transpose(2, 0, 1)
        # print(ori_img.shape)
        return img, mask, {'img_id': img_path}, ori_img
    

"""
CSVBSDatasetV2: Enhanced version with hard sample support
Improvements over V1:
- Added hard sample ratio and hard sample data list
- Introduced hard sample sampling strategy
- Enhanced data balancing with four types:
  1. Unlabeled negative data (neg_ratio)
  2. Labeled pure background (pure_bg_ratio)
  3. Mixed labeled data (mix_ratio)
  4. Labeled foreground with hard samples (pos_ratio = 1 - neg_ratio - pure_bg_ratio - mix_ratio)
- Better handling of white regions in images
"""

class CSVBSDatasetV2(torch.utils.data.Dataset):
    def __init__(self, data_root_path, 
                 wsi_data_list_path, neg_wsi_data_list_path, hard_sample_data_list,
                 wsi_data_details_info_dir=None,
                 img_ext='.png', mask_ext='.png', target_patch_num=5000,
                 target_image_size=2048, 
                 random_image_size=[1024, 3072], random_size_ratio=0.2,
                 no_label_neg_ratio=0.3, no_label_neg_data_root_path=None,
                 pure_bg_ratio=0.15, mix_ratio=0.15, hard_sample_ratio=0.1, ignore_value=2,  
                 transform=None, preprocessing=None, cust_trans=False,
                 try_limit=100):

        self.img_dir = os.path.join(data_root_path, 'img')
        self.mask_dir = os.path.join(data_root_path, 'mask')
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.try_limit = try_limit

        self.target_path_num = target_patch_num
        self.target_img_size = target_image_size
        self.pure_bg_ratio = pure_bg_ratio

        # image process
        self.transform = transform
        self.preprocessing = preprocessing
        self.cust_trans = cust_trans

        # random crop setting
        self.random_size_ratio = random_size_ratio
        self.random_image_size = random_image_size

        with open(wsi_data_list_path, 'r') as f:
            self.wsi_data_list = f.read().splitlines() 

        self.wsi_data_list = [f.rstrip() for f in self.wsi_data_list]


        # process fg files
        self.hard_sample_data_list = []
        self.fg_wsi_file_list = []
        self.fg_wsi_file_details_dict = {}
        fg_detials_path = os.path.join(wsi_data_details_info_dir, 'fg')
        fg_details_info_file_list = scan_files(fg_detials_path, replace_root=False)
        for tmp_f_path in fg_details_info_file_list:
            tmp_wsi_name = os.path.splitext(os.path.basename(tmp_f_path))[0]            
            
            if tmp_wsi_name in self.wsi_data_list:        
                with open(tmp_f_path, 'r') as f:
                    tmp_details_file_list = f.read().splitlines()         
                self.fg_wsi_file_list.append(tmp_wsi_name)
                self.fg_wsi_file_details_dict[tmp_wsi_name] = tmp_details_file_list
                if tmp_wsi_name in hard_sample_data_list:
                    self.hard_sample_data_list.append(tmp_wsi_name)


        # process bg files
        self.bg_wsi_file_list = []
        self.bg_wsi_file_details_dict = {}
        bg_detials_path = os.path.join(wsi_data_details_info_dir, 'no_fg')
        bg_details_info_file_list = scan_files(bg_detials_path, replace_root=False)
        for tmp_f_path in bg_details_info_file_list:
            tmp_wsi_name = os.path.splitext(os.path.basename(tmp_f_path))[0]            
            
            if tmp_wsi_name in self.wsi_data_list:        
                with open(tmp_f_path, 'r') as f:
                    tmp_details_file_list = f.read().splitlines()         
                self.bg_wsi_file_list.append(tmp_wsi_name)
                self.bg_wsi_file_details_dict[tmp_wsi_name] = tmp_details_file_list


        # merge all details
        wsi_file_details_dict = {}
        for tmp_wsi_f in self.wsi_data_list:     
            tmp_file_details_list = []       
            if tmp_wsi_f in self.fg_wsi_file_details_dict:
                tmp_file_details_list.extend(self.fg_wsi_file_details_dict[tmp_wsi_f])

            if tmp_wsi_f in self.bg_wsi_file_details_dict:
                tmp_file_details_list.extend(self.bg_wsi_file_details_dict[tmp_wsi_f])
    
            if len(tmp_file_details_list) > 0:
                wsi_file_details_dict[tmp_wsi_f] = tmp_file_details_list

        self.wsi_file_details_dict = wsi_file_details_dict

        # process unlabeled pure neg data
        with open(neg_wsi_data_list_path, 'r') as f:
            self.neg_wsi_data_list = f.read().splitlines() 

        self.neg_img_dir = os.path.join(no_label_neg_data_root_path, 'img')
        self.neg_mask_dir = os.path.join(no_label_neg_data_root_path, 'mask')
        self.neg_ratio = no_label_neg_ratio

        self.neg_loc_name_dict = {}
        final_valid_neg_wsi_data_list = []
        for tmp_wsi in self.neg_wsi_data_list:
            tmp_img_path = os.path.join(self.neg_img_dir, tmp_wsi)
            tmp_details_img_list = scan_files(tmp_img_path, ext_list=['.png'], replace_root=True)
            if len(tmp_details_img_list) > 0 :
                self.neg_loc_name_dict[tmp_wsi] = tmp_details_img_list
                final_valid_neg_wsi_data_list.append(tmp_wsi)
        self.neg_wsi_data_list = final_valid_neg_wsi_data_list

        self.mix_ratio = mix_ratio
        self.ignore_value = ignore_value
        self.pos_ratio = 1.-(self.neg_ratio+self.pure_bg_ratio+self.mix_ratio)
        self.hard_sample_ratio = hard_sample_ratio
        

            
    def __len__(self):
        return self.target_path_num

    def __getitem__(self, idx):

        current_type = random.choices([0,1,2,3], 
                       weights=[self.neg_ratio, 
                                self.pure_bg_ratio,
                                self.mix_ratio,
                                self.pos_ratio])[0]
        # 0: no label neg, which from cropped bg image, no random size
        # 1: labeled pure bg
        # 2: labeled data include bg and fg
        # 3: labeled fg 

        if current_type == 0:
            target_wsi = random.choice(self.neg_wsi_data_list)
            tmp_loc = random.choice(self.neg_loc_name_dict[target_wsi])
            img_path = os.path.join(self.neg_img_dir, target_wsi, tmp_loc)
            mask_path = os.path.join(self.neg_mask_dir, target_wsi, tmp_loc)

            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = np.array(Image.open(mask_path))

        else:  
            break_loop = True
            try_count = 0
            
            while True:
                if current_type == 1:
                    target_wsi = random.choice(self.bg_wsi_file_list)
                    tmp_loc = random.choice(self.bg_wsi_file_details_dict[target_wsi])
                    img_path = os.path.join(self.img_dir, target_wsi, tmp_loc)
                    mask_path = os.path.join(self.mask_dir, target_wsi, tmp_loc)            
                elif current_type == 2:
                    target_wsi = random.choice(self.wsi_data_list)
                    tmp_loc = random.choice(self.wsi_file_details_dict[target_wsi])
                    img_path = os.path.join(self.img_dir, target_wsi, tmp_loc)
                    mask_path = os.path.join(self.mask_dir, target_wsi, tmp_loc)                
                else:
                    if_hs = random.choices([0, 1], weights=[1-self.hard_sample_ratio, self.hard_sample_ratio])[0]
                    
                    if if_hs == 1:
                        target_wsi = random.choice(self.hard_sample_data_list)
                    else:
                        target_wsi = random.choice(self.fg_wsi_file_list)



                    tmp_loc = random.choice(self.fg_wsi_file_details_dict[target_wsi])
                    img_path = os.path.join(self.img_dir, target_wsi, tmp_loc)
                    mask_path = os.path.join(self.mask_dir, target_wsi, tmp_loc)        

                whole_img_array = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                whole_mask_array = load_png(mask_path)

                h,w = whole_img_array.shape[:2]

                current_target_size = min(h,w)



                # process random crop
                if random.random() < self.random_size_ratio:
                    crop_size = random.randint(min(self.random_image_size[0], current_target_size), 
                                            min(self.random_image_size[1], current_target_size))
                else:
                    crop_size = self.target_img_size

                y_range = h - crop_size
                x_range = w - crop_size

                start_x = random.randint(0, max(0,x_range-1))
                start_y = random.randint(0, max(0,y_range-1))
                end_x = start_x + crop_size
                end_y = start_y + crop_size

                patch_mask_array = np.array(whole_mask_array[start_y:end_y, start_x:end_x])
                
                img = np.array(whole_img_array[start_y:end_y, start_x:end_x])

                mask = patch_mask_array


                check_img = np.array(img)
                r_check_mask = check_img[:,:,0] > 220
                g_check_mask = check_img[:,:,1] > 220
                b_check_mask = check_img[:,:,2] > 220
                white_ignore_mask = r_check_mask&g_check_mask&b_check_mask

                mask = np.array(mask)
                fg_mask = mask == 1
                mask[(white_ignore_mask & fg_mask)] = self.ignore_value

                if current_type == 3:
                    num_pos = np.sum(mask==1)
                    pos_ratio = num_pos/(crop_size*crop_size)
                    # print('### debug type 3', current_type)
                    # print(num_pos, pos_ratio)
                    if pos_ratio > 0.05:
                        break_loop = True
                    else:
                        break_loop = False
                else:
                    break_loop = True

                try_count += 1

                if break_loop or try_count > self.try_limit:
                    break

        # generate random crop done
        ori_img = np.array(img)


        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
        else:
            img = img.astype('float32') / 255
            img = img.transpose(2, 0, 1)


        mask = mask.astype('float32')
        
        # print('### debug mask')
        # print(np.unique(mask, return_counts=True))

        ori_img = cv2.resize(ori_img, (img.shape[2], img.shape[1]) )
        ori_img = ori_img.transpose(2, 0, 1)
        # print(ori_img.shape)
        return img, mask, {'img_id': img_path}, ori_img


"""
CSVBSDatasetV3: Advanced version with improved sampling and cropping
Improvements over V2:
- Replaced hard sample list with hard sample mask directory
- Introduced overlap-based random cropping
- Enhanced region selection using reference masks
- More precise positive sample selection
- Better handling of hard samples with dedicated mask loading
- Improved data augmentation pipeline
Key features:
- Overlap-based cropping for better boundary handling
- Reference mask-based region selection
- Dedicated hard sample mask loading
- More flexible data sampling strategy
Data distribution:
- Unlabeled negative data (neg_ratio)
- Labeled pure background (pure_bg_ratio)
- Mixed labeled data (mix_ratio)
- Labeled foreground with hard samples (pos_ratio = 1 - neg_ratio - pure_bg_ratio - mix_ratio)
"""

class CSVBSDatasetV3(torch.utils.data.Dataset):
    def __init__(self, data_root_path, 
                 wsi_data_list_path, neg_wsi_data_list_path, hard_sample_mask_dir,
                 wsi_data_details_info_dir=None,
                 img_ext='.png', mask_ext='.png', target_patch_num=5000,
                 target_image_size=2048, 
                 random_image_size=[1024, 3072], random_size_ratio=0.2,
                 no_label_neg_ratio=0.3, no_label_neg_data_root_path=None,
                 pure_bg_ratio=0.15, mix_ratio=0.15, hard_sample_ratio=0.1, ignore_value=2,  
                 transform=None, preprocessing=None, cust_trans=False,
                 try_limit=100):

        self.img_dir = os.path.join(data_root_path, 'img')
        self.mask_dir = os.path.join(data_root_path, 'mask')
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.try_limit = try_limit

        self.target_path_num = target_patch_num
        self.target_img_size = target_image_size
        self.pure_bg_ratio = pure_bg_ratio

        # image process
        self.transform = transform
        self.preprocessing = preprocessing
        self.cust_trans = cust_trans

        # random crop setting
        self.random_size_ratio = random_size_ratio
        self.random_image_size = random_image_size

        # hard sample mask dir
        self.hard_sample_mask_dir = hard_sample_mask_dir

        with open(wsi_data_list_path, 'r') as f:
            self.wsi_data_list = f.read().splitlines() 

        self.wsi_data_list = [f.rstrip() for f in self.wsi_data_list]

        hard_sample_data_list = os.listdir(hard_sample_mask_dir)

        # process fg files
        self.hard_sample_data_list = []
        self.fg_wsi_file_list = []
        self.fg_wsi_file_details_dict = {}
        fg_detials_path = os.path.join(wsi_data_details_info_dir, 'fg')
        fg_details_info_file_list = scan_files(fg_detials_path, replace_root=False)
        for tmp_f_path in fg_details_info_file_list:
            tmp_wsi_name = os.path.splitext(os.path.basename(tmp_f_path))[0]            
            
            if tmp_wsi_name in self.wsi_data_list:        
                with open(tmp_f_path, 'r') as f:
                    tmp_details_file_list = f.read().splitlines()         
                self.fg_wsi_file_list.append(tmp_wsi_name)
                self.fg_wsi_file_details_dict[tmp_wsi_name] = tmp_details_file_list
                if tmp_wsi_name in hard_sample_data_list:
                    self.hard_sample_data_list.append(tmp_wsi_name)

        # process hard sample files
        self.hard_sample_data_dict = {}
        for tmp_wsi_f in hard_sample_data_list:
            tmp_mask_path = os.path.join(hard_sample_mask_dir, tmp_wsi_f)
            tmp_files_list = scan_files(tmp_mask_path, ext_list=['.png'], replace_root=False)
            self.hard_sample_data_dict[tmp_wsi_f] = tmp_files_list
            



        # process bg files
        self.bg_wsi_file_list = []
        self.bg_wsi_file_details_dict = {}
        bg_detials_path = os.path.join(wsi_data_details_info_dir, 'no_fg')
        bg_details_info_file_list = scan_files(bg_detials_path, replace_root=False)
        for tmp_f_path in bg_details_info_file_list:
            tmp_wsi_name = os.path.splitext(os.path.basename(tmp_f_path))[0]            
            
            if tmp_wsi_name in self.wsi_data_list:        
                with open(tmp_f_path, 'r') as f:
                    tmp_details_file_list = f.read().splitlines()         
                self.bg_wsi_file_list.append(tmp_wsi_name)
                self.bg_wsi_file_details_dict[tmp_wsi_name] = tmp_details_file_list


        # merge all details
        wsi_file_details_dict = {}
        for tmp_wsi_f in self.wsi_data_list:     
            tmp_file_details_list = []       
            if tmp_wsi_f in self.fg_wsi_file_details_dict:
                tmp_file_details_list.extend(self.fg_wsi_file_details_dict[tmp_wsi_f])

            if tmp_wsi_f in self.bg_wsi_file_details_dict:
                tmp_file_details_list.extend(self.bg_wsi_file_details_dict[tmp_wsi_f])
    
            if len(tmp_file_details_list) > 0:
                wsi_file_details_dict[tmp_wsi_f] = tmp_file_details_list

        self.wsi_file_details_dict = wsi_file_details_dict

        # process unlabeled pure neg data
        with open(neg_wsi_data_list_path, 'r') as f:
            self.neg_wsi_data_list = f.read().splitlines() 

        self.neg_img_dir = os.path.join(no_label_neg_data_root_path, 'img')
        self.neg_mask_dir = os.path.join(no_label_neg_data_root_path, 'mask')
        self.neg_ratio = no_label_neg_ratio

        self.neg_loc_name_dict = {}
        final_valid_neg_wsi_data_list = []
        for tmp_wsi in self.neg_wsi_data_list:
            tmp_img_path = os.path.join(self.neg_img_dir, tmp_wsi)
            tmp_details_img_list = scan_files(tmp_img_path, ext_list=['.png'], replace_root=True)
            if len(tmp_details_img_list) > 0 :
                self.neg_loc_name_dict[tmp_wsi] = tmp_details_img_list
                final_valid_neg_wsi_data_list.append(tmp_wsi)
        self.neg_wsi_data_list = final_valid_neg_wsi_data_list

        self.mix_ratio = mix_ratio
        self.ignore_value = ignore_value
        self.pos_ratio = 1.-(self.neg_ratio+self.pure_bg_ratio+self.mix_ratio)
        self.hard_sample_ratio = hard_sample_ratio
        

            
    def __len__(self):
        return self.target_path_num

    def __getitem__(self, idx):

        current_type = random.choices([0,1,2,3], 
                       weights=[self.neg_ratio, 
                                self.pure_bg_ratio,
                                self.mix_ratio,
                                self.pos_ratio])[0]
        # 0: no label neg, which from cropped bg image, no random size
        # 1: labeled pure bg
        # 2: labeled data include bg and fg
        # 3: labeled fg 

        if current_type == 0:
            target_wsi = random.choice(self.neg_wsi_data_list)
            tmp_loc = random.choice(self.neg_loc_name_dict[target_wsi])
            img_path = os.path.join(self.neg_img_dir, target_wsi, tmp_loc)
            mask_path = os.path.join(self.neg_mask_dir, target_wsi, tmp_loc)

            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = np.array(Image.open(mask_path))

        else:  
            break_loop = True
            try_count = 0
            
            while True:
                if current_type == 1:
                    target_wsi = random.choice(self.bg_wsi_file_list)
                    tmp_loc = random.choice(self.bg_wsi_file_details_dict[target_wsi])
                    img_path = os.path.join(self.img_dir, target_wsi, tmp_loc)
                    mask_path = os.path.join(self.mask_dir, target_wsi, tmp_loc)            
                elif current_type == 2:
                    target_wsi = random.choice(self.wsi_data_list)
                    tmp_loc = random.choice(self.wsi_file_details_dict[target_wsi])
                    img_path = os.path.join(self.img_dir, target_wsi, tmp_loc)
                    mask_path = os.path.join(self.mask_dir, target_wsi, tmp_loc)                
                else:
                    if_hs = random.choices([0, 1], weights=[1-self.hard_sample_ratio, self.hard_sample_ratio])[0]
                    
                    if if_hs == 1:
                        target_wsi = random.choice(self.hard_sample_data_list)
                        tmp_loc = random.choice(self.hard_sample_data_dict[target_wsi])
                    else:
                        target_wsi = random.choice(self.fg_wsi_file_list)
                        tmp_loc = random.choice(self.fg_wsi_file_details_dict[target_wsi])

                    img_path = os.path.join(self.img_dir, target_wsi, tmp_loc)
                    mask_path = os.path.join(self.mask_dir, target_wsi, tmp_loc)



                whole_img_array = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                whole_mask_array = load_png(mask_path)

                h,w = whole_img_array.shape[:2]

                current_target_size = min(h,w)

                # process random crop size
                if random.random() < self.random_size_ratio:
                    crop_size = random.randint(min(self.random_image_size[0], current_target_size), 
                                            min(self.random_image_size[1], current_target_size))
                else:
                    crop_size = self.target_img_size

                crop_size = min(crop_size, current_target_size)

                # generate random crop position
                overlap_size = random.randint(0, int(crop_size*0.9))

                if current_type == 3 and if_hs == 1:
                    ref_mask_path = os.path.join(self.hard_sample_mask_dir, target_wsi, tmp_loc)
                    ref_mask_array = load_png(ref_mask_path)
                elif current_type == 3:
                    ref_mask_array = whole_mask_array == 1
                else:
                    ref_mask_array = None

                regions = get_regions((h,w), crop_size, overlap_size, ref_mask_array)

                start_x, start_y, end_x, end_y = random.choice(regions)

                # read patch image and mask
                patch_mask_array = np.array(whole_mask_array[start_y:end_y, start_x:end_x])
                
                img = np.array(whole_img_array[start_y:end_y, start_x:end_x])

                mask = patch_mask_array


                check_img = np.array(img)
                r_check_mask = check_img[:,:,0] > 220
                g_check_mask = check_img[:,:,1] > 220
                b_check_mask = check_img[:,:,2] > 220
                white_ignore_mask = r_check_mask&g_check_mask&b_check_mask

                mask = np.array(mask)
                fg_mask = mask == 1
                mask[(white_ignore_mask & fg_mask)] = self.ignore_value

                if current_type == 3:
                    num_pos = np.sum(mask==1)

                    if num_pos > 0:
                        break_loop = True
                    else:
                        break_loop = False
                else:
                    break_loop = True

                try_count += 1

                if break_loop or try_count > self.try_limit:
                    break



        # generate random crop done
        ori_img = np.array(img)



        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
        else:
            img = img.astype('float32') / 255
            img = img.transpose(2, 0, 1)


        mask = mask.astype('float32')
        


        ori_img = cv2.resize(ori_img, (img.shape[2], img.shape[1]) )
        ori_img = ori_img.transpose(2, 0, 1)
        # print(ori_img.shape)
        return img, mask, {'img_id': img_path}, ori_img

