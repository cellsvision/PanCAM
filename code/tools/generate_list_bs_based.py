import numpy as np
import cv2
import pandas as pd
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def scan_files(input_file_path, ext_list = ['.txt'], replace_root=True):
    file_list = []
    for root, dirs, files in os.walk(input_file_path):
        # scan all files and put it into list

        for f in files:
            if os.path.splitext(f)[1].lower() in ext_list:
                if replace_root == True:
                    file_list.append(os.path.join(root, f).replace("\\","/").replace(os.path.join(input_file_path, "").replace("\\","/"), "", 1 ))
                elif replace_root:
                    result_path = os.path.join(root, f).replace("\\","/").replace(os.path.join(replace_root, "").replace("\\","/"), "", 1 )
                    file_list.append(result_path)
                else:
                    file_list.append(os.path.join(root, f).replace("\\","/"))
    return file_list


def write_list(output_path, list_content):
    if not os.path.isdir(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        for tpath in list_content:
            f.write(tpath + '\n')




input_data_path = 'path_to/input_data_path'


output_list_path = 'path_to/output_list_path'


# output include:
# 1. labeled_data_list.txt:
# 1.1. each <wsi>.txt contain the file which include lsil
# 1.2. each <wsi>.txt contain the file which not include lsil
# 2. unlabeled_data_list.txt:
# 3. train_list.txt
# 4. val_list.txt


# step 1: labeled_data process
labeled_mask_dir = os.path.join(input_data_path, 'mask')
labeled_img_dir = os.path.join(input_data_path, 'img')


output_wsi_details_dir = os.path.join(output_list_path, 'details_info')

labeled_wsi_file_list = os.listdir(labeled_mask_dir)
print(labeled_wsi_file_list[:10])
print(len(labeled_wsi_file_list))

total_lsil_file_list = [] 
total_no_lsil_file_list = [] 

output_label_data_list = os.path.join(output_list_path, 'labeled.txt')
write_list(output_label_data_list, labeled_wsi_file_list)

print('### label wsi file Num:', len(labeled_wsi_file_list))


for tmp_wsi in labeled_wsi_file_list:
    current_mask_dir = os.path.join(labeled_mask_dir, tmp_wsi)
    files_list = scan_files(current_mask_dir, ext_list=['.png'], replace_root=False)
    lsil_file_list = []
    no_lsil_file_list = []
    
    for f in files_list:
        print('try to open {}'.format(f))
        tmp_mask = np.array(Image.open(f))
        value, value_count = np.unique(tmp_mask, return_counts=True)
        if 1 in value:
            lsil_file_list.append(os.path.basename(f))
        else:
            no_lsil_file_list.append(os.path.basename(f))

    total_lsil_file_list.extend(lsil_file_list)
    total_no_lsil_file_list.extend(no_lsil_file_list)   

    print('### debug wsi count')
    print(tmp_wsi)
    print(len(files_list))
    print(len(lsil_file_list))
    print(lsil_file_list[:10])
    print(len(no_lsil_file_list))
    print('### debug wsi count done')
    output_wsi_details_lsil_path = os.path.join(output_wsi_details_dir, 'fg', tmp_wsi+'.txt')
    output_wsi_details_no_lsil_path = os.path.join(output_wsi_details_dir, 'no_fg', tmp_wsi+'.txt')

    if len(lsil_file_list) > 0:
        write_list(output_wsi_details_lsil_path, lsil_file_list)
    if len(no_lsil_file_list) > 0:
        write_list(output_wsi_details_no_lsil_path, no_lsil_file_list)

print('### files summary ###:')
# print(len(total_file_list))
print(len(total_lsil_file_list))
print(len(total_no_lsil_file_list))

