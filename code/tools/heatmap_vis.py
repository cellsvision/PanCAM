import os
import cv2
import numpy as np
import pickle

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

if __name__ == '__main__':

    input_img_dir = 'Path_To_Image_Dir'
    input_heatmap_dir = 'Path_To_Heatmap_Dir'
    output_blend_dir = 'Path_To_Output_Dir'

    img_files_list = scan_files(input_img_dir, ext_list=['.png'], replace_root=True)

    heatmap_files_list = [os.path.splitext(f)[0]+'.heatmap' for f in img_files_list]


    for tmp_img, tmp_heatmap in zip(img_files_list, heatmap_files_list):
        img = cv2.imread(os.path.join(input_img_dir, tmp_img))
        
        with open(os.path.join(input_heatmap_dir, tmp_heatmap), 'rb') as f:
            print('load heatmap from: ', os.path.join(input_heatmap_dir, tmp_heatmap))
            heatmap = pickle.load(f)
        
        heatmap_img = (heatmap*255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap_img, 2)

        blend_img = cv2.addWeighted(img, 0.5, heatmap_img, 0.5, 0)
        output_blend_img_path = os.path.join(output_blend_dir, os.path.splitext(tmp_img)[0]+'.jpg')

        os.makedirs(os.path.dirname(output_blend_img_path), exist_ok=True)
        print('save blend img to: ', output_blend_img_path)
        cv2.imwrite(output_blend_img_path, blend_img)