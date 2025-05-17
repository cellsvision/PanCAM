import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from albumentations.augmentations import transforms, geometric
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import numpy as np

import archs
from dataset import Dataset
from metrics import iou_score_per_class
from utils import AverageMeter, get_preprocessing

import segmentation_models_pytorch as smp

from params_list import ARCH_DICT, LOSS_DICT


def scan_files(input_file_path, ext_list):
    file_list = []
    for root, dirs, files in os.walk(input_file_path):
        # scan all files and put it into list

        for f in files:
            if os.path.splitext(f)[1].lower() in ext_list:
                file_list.append(os.path.join(root, f).replace("\\","/").replace(os.path.join(input_file_path, "").replace("\\","/"), "", 1 ))

    return file_list

def save_blend(out_merged_path, image_array, current_mask, alpha=.5):
    mask_img = Image.fromarray(current_mask)
    mask_img = mask_img.convert("RGB")

    pil_img = Image.fromarray(image_array[:, :, ::-1])
    blend_img = Image.blend(mask_img, pil_img, alpha=alpha)
    blend_img.save(out_merged_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data_path',
                        help='input_data_path')
    parser.add_argument('--dataset_name', default=None,
                        help='dataset name')
    parser.add_argument('--name', default=None,
                        help='model name')
    parser.add_argument('--epoch', default=None,
                        type=int, help='model epoch number')
    parser.add_argument('--output_data_path', default=None,
                        help='output result path')
    parser.add_argument('--class_name', default='nlst,non_nlst,others,cn',
                        help='class name list')
    parser.add_argument('--test_size', default=1., type=float)
    parser.add_argument('--random_state', default=41, type=int)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    input_data_path = args.input_data_path
    output_data_path = args.output_data_path
    class_name_list = args.class_name.rstrip().split(',')
    dataset_name = None if args.dataset_name == None else args.dataset_name
    test_size = args.test_size
    random_state = args.random_state
    epoch = args.epoch

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if len(class_name_list) != config['num_classes']:
        raise Exception('num of class name not equal to num_classes in config file.')

    print("######### current epoch:{}".format(str(epoch)))

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])


    model = ARCH_DICT[config['arch']](
        encoder_name=config['encoder'],
        encoder_weights=config['encoder_weight'],
        classes=config['num_classes'],
        activation=config['act'],
    )



    if torch.cuda.device_count() > 1:
        print('number of GPU > 1, using data parallel')
        model = nn.DataParallel(model)

    model = model.cuda()

    # Data loading code
    img_ids = scan_files(os.path.join(input_data_path, dataset_name, 'images'), [config['img_ext']])

    img_ids = [os.path.splitext(p)[0] for p in img_ids]
    print(len(img_ids))

    if test_size >= 1 or test_size <= 0:
        val_img_ids = img_ids
    else:
        _, val_img_ids = train_test_split(img_ids, test_size=test_size, random_state=random_state)

    if epoch is None:
        model.load_state_dict(torch.load('models/%s/model.pth' %
                                         config['name']))
    else:
        model.load_state_dict(torch.load('models/%s/%s.pth' %
                                         (config['name'],str(epoch))))
    model.eval()


    if config['encoder_weight'] is not None:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(config['encoder'], config['encoder_weight'])
    else:
        preprocessing_fn = None


    val_transform = Compose([
        #transforms.Resize(config['input_h'], config['input_w']),
        geometric.resize.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(input_data_path, dataset_name, 'images'),
        mask_dir=os.path.join(input_data_path, dataset_name, 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform,
        preprocessing=get_preprocessing(preprocessing_fn),
        mode=config['mode'])
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()
    per_cls_avg_meter = [AverageMeter() for _ in range(config['num_classes'])]



    with torch.no_grad():
        for ori_input, target, meta in tqdm(val_loader, total=len(val_loader)):

            input = ori_input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou, iou_per_cls = iou_score_per_class(output, target)
            avg_meter.update(iou, input.size(0))
            for tmp_m, tmp_v in zip(per_cls_avg_meter, iou_per_cls):
                tmp_m.update(tmp_v, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()


            if output_data_path is not None:
                for i in range(len(output)):
                    for c in range(config['num_classes']):

                        out_mask = os.path.join(output_data_path, config['name'], 'masks', class_name_list[c], meta['img_id'][i] + '.jpg')
                        os.makedirs(os.path.dirname(out_mask), exist_ok=True)
                        cv2.imwrite(out_mask, (output[i, c] * 255).astype('uint8'))

                        ori_img = cv2.imread(os.path.join(input_data_path, dataset_name, 'images', meta['img_id'][i] + config['img_ext']))
                        ori_img = cv2.resize(ori_img, (config['input_w'], config['input_h']))
                        out_blend = os.path.join(output_data_path, config['name'], 'blend', class_name_list[c], meta['img_id'][i] + '.jpg')
                        os.makedirs(os.path.dirname(out_blend), exist_ok=True)
                        save_blend(out_blend, ori_img, (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meter.avg)
    for i, cls_name in enumerate(class_name_list):
        print('{} IoU: {:04f}'.format(cls_name, per_cls_avg_meter[i].avg))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
