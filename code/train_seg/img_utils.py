import cv2
from PIL import Image
import numpy as np

# from .utils import save_pkl

def draw_single_contour_to_image(input_image, contour, color_filled, color_border=None):
    contours = np.reshape(contour, (1, -1, 1, 2))
    contours = contours.astype(np.int32)

    new_image = np.array(input_image)
    cv2.drawContours(new_image, contours, -1, (color_filled), -1)

    if color_border:
        cv2.drawContours(new_image, contours, -1, (color_border), 5)

    return new_image


def create_mask(patch_area, all_seg_dict, label_list, crop_size, init_type=0, ignore_label_index=255):
    mask = np.full((crop_size, crop_size, 1), init_type, dtype=np.uint8)
    x1, y1, x2, y2 = patch_area

    for tmp_color_index in range(len(label_list) - 1):
        for tmp_label_index in range(len(label_list[tmp_color_index])):
            # print(label_list[tmp_color_index])
            # print(label_list[tmp_color_index][tmp_label_index])
            # print(all_seg_dict.keys())
            if label_list[tmp_color_index][tmp_label_index] in all_seg_dict:
                for points_array in all_seg_dict[label_list[tmp_color_index][tmp_label_index]]:
                    contour = np.array(points_array) - (x1, y1)
                    min_x, min_y = np.min(contour[:, 0]), np.min(contour[:, 1])
                    max_x, max_y = np.max(contour[:, 0]), np.max(contour[:, 1])
                    if max_x <= 0 or max_y <= 0 or min_x >= crop_size or min_y >= crop_size:
                        continue
                    mask = draw_single_contour_to_image(mask, contour, tmp_color_index, color_border=ignore_label_index)

    return mask

def create_mask_split(patch_area, all_seg_dict, label_list, crop_size, init_type=0, ignore_label_index=255):
    valid_label_list = [x for x in label_list if x is not '']

    mask = np.full((len(valid_label_list) - 1, crop_size, crop_size), init_type, dtype=np.uint8)
    x1, y1, x2, y2 = patch_area

    for tmp_color_index in range(len(label_list) - 1):
        for tmp_label_index in range(len(label_list[tmp_color_index])):
            # print(label_list[tmp_color_index])
            # print(label_list[tmp_color_index][tmp_label_index])
            # print(all_seg_dict.keys())
            if label_list[tmp_color_index][tmp_label_index] in all_seg_dict:
                for points_array in all_seg_dict[label_list[tmp_color_index][tmp_label_index]]:
                    contour = np.array(points_array) - (x1, y1)
                    min_x, min_y = np.min(contour[:, 0]), np.min(contour[:, 1])
                    max_x, max_y = np.max(contour[:, 0]), np.max(contour[:, 1])
                    if max_x <= 0 or max_y <= 0 or min_x >= crop_size or min_y >= crop_size:
                        continue
                    mask[tmp_color_index] = draw_single_contour_to_image(mask[tmp_color_index], contour, 255)
    mask = mask[1:] # remove bg
    mask = np.transpose(mask, (1, 2, 0))
    # print("debug in img_utils")
    # print(mask.shape)
    return mask


def save_img(out_image_path, image_array):
    #print(image_array)
    cv2.imwrite(out_image_path, image_array)


def load_mask(in_mask_path):
    mask_img = np.array(Image.open(in_mask_path))
    return mask_img



def save_mask(out_mask_path, current_mask, palette_list):
    # print("### debug save mask")
    # print(current_mask.shape)
    # print(len(current_mask.shape))
    # print(current_mask.dtype)
    if len(current_mask.shape) == 3:
        pil_mat = np.squeeze(current_mask, axis=2)
    else:
        pil_mat = current_mask
    pil_img = Image.fromarray(pil_mat)
    pil_img = pil_img.convert("P")
    pil_img.putpalette(palette_list)
    pil_img.save(out_mask_path)

# def save_mask_pkl(out_mask_path, masks):
#     save_pkl(masks, out_mask_path)

def save_blend_split(out_merged_path, image_array, current_masks, palette_list, alpha=.5):
    mask_mat = np.zeros(current_masks.shape[:2])
    new_masks = np.transpose(current_masks, (2, 0, 1))
    # print('### debug in save_blend 0')
    # print(new_masks.shape[0])
    # from collections import Counter
    for index in range(new_masks.shape[0]):
        mask_mat[new_masks[index]==255] = (index+1)
    #     print(Counter(mask_mat.flatten()))
    # print('### debug in save_blend 0.5')
    # print(Counter(mask_mat.flatten()))
    # print('### debug in save_blend 1')
    mask_img = Image.fromarray(mask_mat)
    mask_img = mask_img.convert("P")
    mask_img.putpalette(palette_list)
    mask_img = mask_img.convert("RGB")
    # print('### debug in save_blend done')

    pil_img = Image.fromarray(image_array[:, :, ::-1])
    blend_img = Image.blend(mask_img, pil_img, alpha=alpha)
    blend_img.save(out_merged_path)



def save_blend(out_merged_path, image_array, current_mask, palette_list, alpha=.5):
    mask_mat = np.squeeze(current_mask, axis=2)
    mask_img = Image.fromarray(mask_mat)
    mask_img = mask_img.convert("P")
    mask_img.putpalette(palette_list)
    mask_img = mask_img.convert("RGB")

    pil_img = Image.fromarray(image_array[:, :, ::-1])
    blend_img = Image.blend(mask_img, pil_img, alpha=alpha)
    blend_img.save(out_merged_path)

