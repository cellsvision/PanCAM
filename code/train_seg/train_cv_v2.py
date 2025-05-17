import argparse
from logging import raiseExceptions
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import OrderedDict
from glob import glob
import copy

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler as ampgs
from torch.cuda.amp import autocast as autocast
from torchvision.utils import make_grid, save_image

import yaml
from albumentations.augmentations import geometric, transforms, dropout
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from sklearn.model_selection import RepeatedKFold
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import cv2


import segmentation_models_pytorch as smp


from dataset import CSVBSDatasetV1, CSVBSDatasetV3

from metrics import iou_score_per_class_multiclass, iou_score_per_class_multiclass_v2
from utils import AverageMeter, str2bool
from radam import RAdam, Adan

from segmentation_models_pytorch.losses import *
from segmentation_models_pytorch.scheduler import CosineAnnealingWarmupRestarts

os.environ['TORCH_HOME'] = '/public/tmp/lz_models/models_zoo'

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
              'mcc': MCCLoss,
            }


ARCH_NAMES = ARCH_DICT.keys()
LOSS_NAMES = list(LOSS_DICT.keys())
LOSS_NAMES.append('BCEWithLogitsLoss')
MODE_NAMES = ['binary', 'multiclass', 'multilabel']
ACT_NAMES = [None, 'identity',  'sigmoid', 'softmax2d', 'softmax', 'logsoftmax', 'tanh', 'argmax', 'argmax2d']

import albumentations as albu
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def to_tensor_mask(x, **kwargs):
    return x.astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    if preprocessing_fn is not None:
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor, mask=to_tensor_mask),
        ]
    else:
         _transform = [
            albu.Lambda(image=to_tensor, mask=to_tensor_mask),
        ]
    return albu.Compose(_transform)


def scan_files(input_file_path, ext_list):
    file_list = []
    for root, dirs, files in os.walk(input_file_path):
        # scan all files and put it into list
        for f in files:
            # f = f.decode('ascii')
            if os.path.splitext(f)[1].lower() in ext_list:
                file_list.append(os.path.join(root, f).replace("\\","/").replace(os.path.join(input_file_path, "").replace("\\","/"), "", 1 ))

    return file_list


def add_tb_img(tb_writer, preprocess_params, input_img, ori_img, mask_img, pred_mask, count, prefix_name='img'):
    input_range = preprocess_params['input_range']
    mean = preprocess_params['mean']
    std = preprocess_params['std']

    current_img = input_img.cpu().numpy()

    img_8 = np.ascontiguousarray(current_img)
    img_8[0,:,:] = (img_8[0,:,:]*std[0] + mean[0])*255
    img_8[1,:,:] = (img_8[1,:,:]*std[1] + mean[1])*255
    img_8[2,:,:] = (img_8[2,:,:]*std[2] + mean[2])*255
    img_8[img_8 > 255] = 255
    img_8[img_8 < 0] = 0
    img_8 = img_8.astype(np.uint8)

    img_list = [img_8]


    mask_img = (mask_img).cpu().numpy()

    mask_img = mask_img.astype(np.uint8)
    print('### shape_check 1.0')
    print(mask_img.shape)
    mask_img = np.array(cv2.cvtColor(mask_img,cv2.COLOR_GRAY2RGB)).transpose((2,0,1))
    print('### shape_check 1.1')
    print(mask_img.shape)
    img_list.append(mask_img)

    pred_mask = pred_mask.detach().cpu().numpy()

    pred_mask = np.where(pred_mask>0.5, 255, 0)

    for mi in pred_mask[:]:
        mi = mi.astype(np.uint8)
        mi = np.array(cv2.cvtColor(mi,cv2.COLOR_GRAY2RGB)).transpose((2,0,1))
        print('### shape_check 2')
        print(mi.shape)
        img_list.append(mi)
    img_list = np.array(img_list)


    final_img = make_grid(torch.as_tensor(img_list))


    save_image(final_img, 'debug/{}_{}_result.jpg'.format(prefix_name, str(count)))
    save_image(ori_img, 'debug/{}_{}.jpg'.format(prefix_name, str(count)))



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='unetplusplus',
                        choices=ARCH_NAMES,
                        help='architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: unetplusplus)')
    parser.add_argument('--mode', default='multilabel',
                        choices=MODE_NAMES,
                        help='model: ' +
                        ' | '.join(MODE_NAMES) +
                        ' (default: multilabel)')
    parser.add_argument('--act', default=None,
                        help='activation: ' +
                        ' | '.join(ACT_NAMES[1:]) +
                        ' (default: identity)')
    parser.add_argument('--encoder', default='resnet18',
                        help='encoder name, deafult is resnet18')
    parser.add_argument('--encoder_weight', default=None,
                        help='encoder weight, default is None')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--mix_precision', default=False, type=str2bool)
    parser.add_argument('--show_arch', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')


    # loss
    parser.add_argument('--loss', default='BCEWithLogitsLoss',
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEWithLogitsLoss)\n'+
                        'multiple loss supported: using , to split, e.g. BCEWithLogitsLoss,lovasz')
    parser.add_argument('--loss_weight', default='1.',
                        help='loss weight: ' +
                        'multiple loss weight supported: using , to split, e.g. 0.5,1.')

    # dataset
    parser.add_argument('--labeled_data_root', default=None,
                        help='labeled_data_root')
    parser.add_argument('--l_train_wsi_data_list_path', default=None,
                        help='l_train_wsi_data_list_path')
    parser.add_argument('--l_val_wsi_data_list_path', default=None,
                        help='l_val_wsi_data_list_path')
    parser.add_argument('--wsi_data_details_info_root', default=None,
                        help='wsi_data_details_info_root')

    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')
    parser.add_argument('--base_size', default=512, type=int,
                        help='image base_size')
    parser.add_argument('--crop_size', default=512, type=int,
                        help='image crop_size')
    parser.add_argument('--train_n_sample', default=10000, type=int,
                        help='image train_n_sample')
    parser.add_argument('--val_n_sample', default=1000, type=int,
                        help='image val_n_sample')
    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD', 'RAdam', 'Adan'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD', 'RAdam', 'Adan']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay, similar one used in AdamW (default: 0.02)')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--max_grad_norm', type=float, default=0.0, help='if the l2 norm is large than this hyper-parameter, then we clip the gradient  (default: 0.0, no gradient clip)')
    parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON', help='optimizer epsilon to avoid the bad case where second-order moment is zero (default: None, use opt default 1e-8 in adan)')
    parser.add_argument('--opt_betas', default=[0.98, 0.92, 0.99], type=float, nargs='+', metavar='BETA', help='optimizer betas in Adan (default: None, use opt default [0.98, 0.92, 0.99] in Adan)')
    parser.add_argument('--no_prox', action='store_true', default=False, help='whether perform weight decay like AdamW (default=False)')


    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR', 'CosineAnnealingWarmupRestarts'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--dp_rate', default=-1., type=float)
    parser.add_argument('--class_name', default='nlst,cn',
                        help='class name list')

    parser.add_argument('--tf_log_path', default=None, type=str)
    parser.add_argument('--cv', default=3, type=int)
    parser.add_argument('--n_repeats', default=1, type=int)
    parser.add_argument('--train_data_csv', default=None)
    parser.add_argument('--neg_wsi_data_list_path', default=None)
    parser.add_argument('--neg_data_root_path', default=None)
    # no label bg
    parser.add_argument('--train_neg_ratio', default=0.3, type=float)
    parser.add_argument('--val_neg_ratio', default=0.3, type=float)
    parser.add_argument('--test_neg_ratio', default=0.5, type=float)

    # labeled bg
    parser.add_argument('--train_pure_bg_ratio', default=0.15, type=float)
    parser.add_argument('--val_pure_bg_ratio', default=0.15, type=float)
    parser.add_argument('--test_pure_bg_ratio', default=0.15, type=float)

    # labeled mix data
    parser.add_argument('--train_mix_ratio', default=0.15, type=float)
    parser.add_argument('--val_mix_ratio', default=0.15, type=float)
    parser.add_argument('--test_mix_ratio', default=0.15, type=float)

    # hard sample
    parser.add_argument('--hard_sample_ratio', default=0.1, type=float)
    parser.add_argument('--hard_sample_dir', default=None)

    parser.add_argument('--test_data_csv', default=None)
    parser.add_argument('--pure_bg_test_data_csv', default=None)
    parser.add_argument('--data_csv', default=None)
    parser.add_argument('--cosine_cycle_steps', default=20, type=int)
    parser.add_argument('--cosine_cycle_warmup', default=2, type=int)
    parser.add_argument('--cosine_cycle_gamma', default=0.9, type=float)
    parser.add_argument('--ignore_index',default=None, type=int)
    parser.add_argument('--freeze_layer', default=None, type=str)
    parser.add_argument('--unfreeze_layer', default=None, type=str)
    parser.add_argument('--output_dir', default='models', type=str)
    parser.add_argument('--pretrain_model', default=None, help='pretrain model')

    parser.add_argument('--random_image_size', default=None, help='random_image_size')
    parser.add_argument('--random_size_ratio', default=0.2, type=float, help='pretrain model')



    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer, preprocess_params=None, tb_writer=None, prefix_name='train'):


    per_cls_avg_meter = [AverageMeter() for _ in range(config['num_classes'])]
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'no_bg_iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    count = 0
    for input, target, _, ori_img in train_loader:

        input = input.cuda()

        target = target.cuda()
        

        # compute output
        if config['mix_precision']:
            # raise Exception('mix_precision not supported.')
            scaler = ampgs()

            with autocast():
                if config['deep_supervision']:
                    outputs = model(input)
                    loss = 0
                    for output in outputs:
                        loss += criterion(output, target)
                    loss /= len(outputs)
                    iou, no_bg_iou, iou_per_cls = iou_score_per_class_multiclass_v2(outputs[-1], target, ignore_label=config['ignore_index'])
                    output = outputs[-1]
                else:
                    output = model(input)

                    loss = criterion(output, target)
                    iou, no_bg_iou, iou_per_cls = iou_score_per_class_multiclass_v2(output, target, ignore_label=config['ignore_index'])
            # compute gradient and do optimizing step
            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, no_bg_iou, iou_per_cls = iou_score_per_class_multiclass_v2(outputs[-1], target, ignore_label=config['ignore_index'])
                output = outputs[-1]
            else:
                output = model(input)

                loss = criterion(output, target)
                iou, no_bg_iou, iou_per_cls = iou_score_per_class_multiclass_v2(output, target, ignore_label=config['ignore_index'])

            # compute gradient and do optimizing step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['no_bg_iou'].update(no_bg_iou, input.size(0))


        for tmp_m, tmp_v in zip(per_cls_avg_meter, iou_per_cls):
            tmp_m.update(tmp_v, input.size(0))

        possfix_list = [('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('no_bg_iou', avg_meters['no_bg_iou'].avg),
                        ]

        for i, cls_name in enumerate(config['class_name']):
            possfix_list.append(('iou_{}'.format(cls_name), per_cls_avg_meter[i].avg))


        count+=1


        postfix = OrderedDict(possfix_list)
        pbar.set_postfix(postfix)
        pbar.update(1)

    pbar.close()

    possfix_list = [('loss', avg_meters['loss'].avg),
                    ('iou', avg_meters['iou'].avg),
                    ('no_bg_iou', avg_meters['no_bg_iou'].avg),
                    ]

    for i, cls_name in enumerate(config['class_name']):
        possfix_list.append(('iou_{}'.format(cls_name), per_cls_avg_meter[i].avg))

    return OrderedDict(possfix_list)


def validate(config, val_loader, model, criterion, preprocess_params=None, tb_writer=None, prefix_name='val'):
    per_cls_avg_meter = [AverageMeter() for _ in range(config['num_classes'])]
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'no_bg_iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        count = 0
        for input, target, _, ori_img in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, no_bg_iou, iou_per_cls = iou_score_per_class_multiclass_v2(outputs[-1], target, ignore_label=config['ignore_index'])
                output = outputs[-1]
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, no_bg_iou, iou_per_cls = iou_score_per_class_multiclass_v2(output, target, ignore_label=config['ignore_index'])

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['no_bg_iou'].update(no_bg_iou, input.size(0))

            for tmp_m, tmp_v in zip(per_cls_avg_meter, iou_per_cls):
                tmp_m.update(tmp_v, input.size(0))

            possfix_list = [('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg),
                            ('no_bg_iou', avg_meters['no_bg_iou'].avg),
                            ]

            for i, cls_name in enumerate(config['class_name']):
                possfix_list.append(('iou_{}'.format(cls_name), per_cls_avg_meter[i].avg))


            count+=1

            postfix = OrderedDict(possfix_list)

            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    possfix_list = [('loss', avg_meters['loss'].avg),
                    ('iou', avg_meters['iou'].avg),
                    ('no_bg_iou', avg_meters['no_bg_iou'].avg),
                    ]

    for i, cls_name in enumerate(config['class_name']):
        possfix_list.append(('iou_{}'.format(cls_name), per_cls_avg_meter[i].avg))

    return OrderedDict(possfix_list)



def main():
    config = vars(parse_args())

    config['random_image_size'] = config['random_image_size'].rstrip().split(',')
    config['random_image_size'] = [int(v) for v in config['random_image_size']]

    cv_dataset_list = []

    config['basename'] = copy.deepcopy(config['name'])

    freeze_layer = None if config['freeze_layer'] is None else config['freeze_layer'].split(',')
    unfreeze_layer = None if config['unfreeze_layer'] is None else config['unfreeze_layer'].split(',')


    pretrain_model = config['pretrain_model']
    config['class_name'] = config['class_name'].rstrip().split(',')
    config['loss'] = config['loss'].rstrip().split(',')
    config['loss_weight']= config['loss_weight'].rstrip().split(',')



    if config['act'] == None:
        if config['mode'] == 'multiclass':
            config['act'] = 'softmax2d' # could be None for logits or 'softmax2d' for multicalss segmentation
        else:
            config['act'] = None

    if len(config['class_name']) != config['num_classes']:
        raise Exception('num of class name not equal to num_classes in config file.')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    # config['name'] = config['basename']
    # os.makedirs('models/%s' % config['name'], exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], config['name']), exist_ok=True)


    new_loss_name = ""
    for l in config['loss']:
        new_loss_name += l + '_'
    new_loss_name = new_loss_name[:-1]

    tf_log_path = config['tf_log_path']
    tf_save_name = '{}_{}_{}_{}_{}'.format(
                                        config['name'], config['arch'], new_loss_name,
                                        config['optimizer'], str(config['lr'])
                                        )

    writer = None
    if tf_log_path is not None:
        writer = SummaryWriter(os.path.join(tf_log_path, tf_save_name),comment=tf_save_name)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(os.path.join(config['output_dir'], config['name'],'config.yml'), 'w') as f:
        yaml.dump(config, f)


    if writer is not None:
        config_content = ''
        for k in sorted(config.keys()):
            config_content += '{}: {}\n\n'.format(k, str(config[k]))
        writer.add_text('config:', config_content)


    assert len(config['loss']) == len(config['loss_weight']), \
        'num of loss must equal to num of loss weight, {} vs {}'.format(str(len(config['loss'])), str(len(config['loss_weight'])))

    loss_list = []
    loss_w_list = []
    for loss_name, loss_w in zip(config['loss'], config['loss_weight']):
        if loss_name == 'BCEWithLogitsLoss':
  
            current_criterion = nn.BCEWithLogitsLoss()
        else:
            if config['ignore_index'] == None:
                current_criterion = LOSS_DICT[loss_name](mode=config['mode'])
            else:
                current_criterion = LOSS_DICT[loss_name](mode=config['mode'], ignore_index=config['ignore_index'])
        loss_list.append(current_criterion)
        loss_w_list.append(float(loss_w))

    criterion = JointLoss(loss_list, loss_w_list)

    criterion = criterion.cuda()
    cudnn.benchmark = True

####################################################################################
    # create model


    # create segmentation model with pretrained encoder
    model = ARCH_DICT[config['arch']](
        encoder_name=config['encoder'],
        encoder_weights=config['encoder_weight'],
        classes=config['num_classes'],
        activation=config['act'],
    )

    if config['encoder_weight'] is not None:

        preprocessing_fn, preprocess_params = smp.encoders.get_preprocessing_fn(config['encoder'], config['encoder_weight'])
    else:
        preprocessing_fn = None

    print('### debug encoder weight params')
    print(preprocess_params)


    if torch.cuda.device_count() > 1:
        print('number of GPU > 1, using data parallel')
        multi_gpu = True
        model = nn.DataParallel(model)

    model = model.cuda()

    if pretrain_model is not None:
        model_weight = torch.load(pretrain_model)
        if multi_gpu:
            new_model_weight = OrderedDict()
            for k,v in model_weight.items():
                if 'module.' not in k:
                    name = 'module.'+ k
                else:
                    name = k
                # print(name)
                new_model_weight[name]=v
            model_weight = new_model_weight
        else:
            new_model_weight = OrderedDict()
            for k,v in model_weight.items():
                if 'module.' in k:
                    name = k.replace('module.', '')
                else:
                    name = k
                # print(name)
                new_model_weight[name]=v
            model_weight = new_model_weight

        model.load_state_dict(model_weight)

    

    if config['show_arch']:
        from torchsummary import summary
        summary(model, input_size=(config['input_channels'], config['input_h'], config['input_w']))



    if freeze_layer is not None:
        for m_name, m_param in model.named_parameters():
            for layer_name in freeze_layer:
                if layer_name in m_name:
                    m_param.requires_grad = False

    if unfreeze_layer is not None:
        for m_name, m_param in model.named_parameters():
            for layer_name in unfreeze_layer:
                if layer_name in m_name and m_param.requires_grad==False:
                    m_param.requires_grad = True

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                            nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'RAdam':
        optimizer = RAdam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'Adan':
        optimizer = Adan(params, lr=config['lr'], weight_decay=config['weight_decay'], betas=config['opt_betas'], eps = config['opt_eps'], max_grad_norm=config['max_grad_norm'], no_prox=config['no_prox'])

    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    elif config['scheduler'] == 'CosineAnnealingWarmupRestarts':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=config['cosine_cycle_steps'], max_lr=config['lr'], min_lr=config['min_lr'],
                                                                warmup_steps=config['cosine_cycle_warmup'], gamma=config['cosine_cycle_gamma'])
    else:
        raise NotImplementedError

    # Data loading code
  
    train_transform = Compose([

        geometric.rotate.RandomRotate90(),
        geometric.transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(hue_shift_limit=10),
            transforms.RandomBrightnessContrast(),
            transforms.RGBShift(),

        ], p=1.),
        OneOf([
            transforms.GaussNoise(),
            transforms.ISONoise(),
            transforms.MultiplicativeNoise(),

        ], p=1.),
        OneOf([

            transforms.ChannelShuffle(),
            transforms.ToGray(),
            transforms.ToSepia(),

        ], p=1.),
        dropout.CoarseDropout(min_holes=4, min_height=4, min_width=4, max_holes=30),

        geometric.resize.Resize(config['crop_size'], config['crop_size']),
 
    ])

    val_transform = Compose([
        geometric.resize.Resize(config['crop_size'], config['crop_size']),

    ])


    train_dataset = CSVBSDatasetV3(
                                    data_root_path=config['labeled_data_root'], 
                                    wsi_data_list_path = config['l_train_wsi_data_list_path'],
                                    neg_wsi_data_list_path = config['neg_wsi_data_list_path'],
                                    hard_sample_mask_dir = config['hard_sample_dir'],
                                    hard_sample_ratio = config['hard_sample_ratio'],
                                    wsi_data_details_info_dir = config['wsi_data_details_info_root'],

                                    target_image_size = config['crop_size'],
                                    random_image_size = config['random_image_size'],
                                    random_size_ratio = config['random_size_ratio'],
                                    no_label_neg_ratio = config['train_neg_ratio'],
                                    no_label_neg_data_root_path = config['neg_data_root_path'],
                                    pure_bg_ratio = config['train_pure_bg_ratio'],
                                    mix_ratio = config['train_mix_ratio'] ,

                                    target_patch_num = config['train_n_sample'],
                                    transform = train_transform,
                                    preprocessing=get_preprocessing(preprocessing_fn),
                                )
    val_dataset = CSVBSDatasetV1(
                                    data_root_path=config['labeled_data_root'], 
                                    wsi_data_list_path = config['l_train_wsi_data_list_path'],
                                    neg_wsi_data_list_path= config['neg_wsi_data_list_path'],
                                    wsi_data_details_info_dir = config['wsi_data_details_info_root'],

                                    target_image_size = config['crop_size'],
                                    random_size_ratio = 0,
                                    no_label_neg_ratio = config['val_neg_ratio'],
                                    no_label_neg_data_root_path = config['neg_data_root_path'],
                                    pure_bg_ratio = config['val_pure_bg_ratio'],
                                    mix_ratio = config['val_mix_ratio'] ,
                                    
                                    target_patch_num = int(config['train_n_sample'] * 0.1),
                                    transform = val_transform,
                                    preprocessing=get_preprocessing(preprocessing_fn),
                                )

    test_dataset = CSVBSDatasetV1(  
                                    data_root_path=config['labeled_data_root'], 
                                    wsi_data_list_path = config['l_val_wsi_data_list_path'],
                                    neg_wsi_data_list_path= config['neg_wsi_data_list_path'],
                                    wsi_data_details_info_dir = config['wsi_data_details_info_root'],

                                    target_image_size = config['crop_size'],
                                    random_size_ratio = 0,
                                    no_label_neg_ratio = config['test_neg_ratio'],
                                    no_label_neg_data_root_path = config['neg_data_root_path'],  
                                    pure_bg_ratio = config['test_pure_bg_ratio'],
                                    mix_ratio = config['test_mix_ratio'] ,
                                
                                    target_patch_num = config['val_n_sample'],
                                    transform = val_transform,
                                    preprocessing=get_preprocessing(preprocessing_fn),
                                )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log_list = [
                ('epoch', []),
                ('lr', []),
                ('loss', []),
                ('iou', []),
                ('no_bg_iou', []),
                ('val_loss', []),
                ('val_iou', []),
                ('val_no_bg_iou', []),
                ('test_loss', []),
                ('test_iou', []),
                ('test_no_bg_iou', []),
            ]

    for i, cls_name in enumerate(config['class_name']):
        log_list.append(('iou_{}'.format(cls_name), []))
    for i, cls_name in enumerate(config['class_name']):
        log_list.append(('val_iou_{}'.format(cls_name), []))
    for i, cls_name in enumerate(config['class_name']):
        log_list.append(('test_iou_{}'.format(cls_name), []))

    log = OrderedDict(log_list)

    best_val_iou = 0
    best_test_iou = 0
    best_val_avg_iou = 0
    best_test_avg_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch

        train_log = train(config, train_loader, model, criterion, optimizer, preprocess_params, writer)

        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion, preprocess_params, writer)
        test_log = validate(config, test_loader, model, criterion, preprocess_params, writer, prefix_name='test')

        print('current lr: {:f}'.format(optimizer.param_groups[0]['lr']))
        if config['scheduler'] == 'CosineAnnealingLR' or config['scheduler'] == "CosineAnnealingWarmupRestarts":
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])


        print('loss %.4f - iou %.4f - no_bg_iou %.4f - val_loss %.4f - val_iou %.4f - val_no_bg_iou %.4f - test_loss %.4f - test_iou %.4f - test_no_bg_iou %.4f'
            % (train_log['loss'], train_log['iou'], train_log['no_bg_iou'],
                val_log['loss'], val_log['iou'], val_log['no_bg_iou'],
                test_log['loss'], test_log['iou'], test_log['no_bg_iou'])
                )

        train_msg = ''
        val_msg = ''
        test_msg = ''
        for i, cls_name in enumerate(config['class_name']):
            train_msg += 'train_iou_{} {:04f} - '.format(cls_name, train_log['iou_{}'.format(cls_name)])
        train_msg = train_msg[:-2]
        for i, cls_name in enumerate(config['class_name']):
            val_msg += 'val_iou_{} {:04f} - '.format(cls_name, val_log['iou_{}'.format(cls_name)])
        val_msg = val_msg[:-2]
        for i, cls_name in enumerate(config['class_name']):
            test_msg += 'test_iou_{} {:04f} - '.format(cls_name, test_log['iou_{}'.format(cls_name)])
        test_msg = test_msg[:-2]
        print(train_msg)
        print(val_msg)
        print(test_msg)


        if writer is not None:
            writer.add_scalar('Train_Loss', train_log['loss'], global_step=epoch)
            writer.add_scalar('Val_Loss', val_log['loss'], global_step=epoch)
            writer.add_scalar('Test_Loss', test_log['loss'], global_step=epoch)
            writer.add_scalar('Train_IoU', train_log['iou'], global_step=epoch)
            writer.add_scalar('Val_IoU', val_log['iou'], global_step=epoch)
            writer.add_scalar('Test_IoU', test_log['iou'], global_step=epoch)
            writer.add_scalar('Train_no_bg_IoU', train_log['no_bg_iou'], global_step=epoch)
            writer.add_scalar('Val_no_bg_IoU', val_log['no_bg_iou'], global_step=epoch)
            writer.add_scalar('Test_no_bg_IoU', test_log['no_bg_iou'], global_step=epoch)

            for i, cls_name in enumerate(config['class_name']):
                writer.add_scalar('Train_{}_IoU'.format(cls_name), train_log['iou_{}'.format(cls_name)], global_step=epoch)
            for i, cls_name in enumerate(config['class_name']):
                writer.add_scalar('Val_{}_IoU'.format(cls_name), val_log['iou_{}'.format(cls_name)], global_step=epoch)
            for i, cls_name in enumerate(config['class_name']):
                writer.add_scalar('Test_{}_IoU'.format(cls_name), test_log['iou_{}'.format(cls_name)], global_step=epoch)

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['no_bg_iou'].append(train_log['no_bg_iou'])
        for i, cls_name in enumerate(config['class_name']):
            log['iou_{}'.format(cls_name)].append(train_log['iou_{}'.format(cls_name)])

        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_no_bg_iou'].append(val_log['no_bg_iou'])

        avg_list = []
        for i, cls_name in enumerate(config['class_name']):
            log['val_iou_{}'.format(cls_name)].append(val_log['iou_{}'.format(cls_name)])
            avg_list.append(val_log['iou_{}'.format(cls_name)])
        val_avg_iou = np.average(avg_list)


        log['test_loss'].append(test_log['loss'])
        log['test_iou'].append(test_log['iou'])
        log['test_no_bg_iou'].append(test_log['no_bg_iou'])

        avg_list = []
        for i, cls_name in enumerate(config['class_name']):
            log['test_iou_{}'.format(cls_name)].append(test_log['iou_{}'.format(cls_name)])
            avg_list.append(test_log['iou_{}'.format(cls_name)])
        test_avg_iou = np.average(avg_list)

        pd.DataFrame(log).to_csv(os.path.join(config['output_dir'], config['name'], 'log.csv'), index=False)

        trigger += 1

        if val_log['no_bg_iou'] > best_val_iou:
            torch.save(model.state_dict(), os.path.join(config['output_dir'], config['name'], 'best_nobg_val_iou.pth'))
            best_val_iou = val_log['no_bg_iou']
            print("=> saved best val model")
            trigger = 0

        if test_log['no_bg_iou'] > best_test_iou:
            torch.save(model.state_dict(), os.path.join(config['output_dir'], config['name'], 'best_nobg_test_iou.pth')) 
            best_test_iou = test_log['no_bg_iou']
            print("=> saved best test model")
            trigger = 0


        if val_avg_iou > best_val_avg_iou:
            torch.save(model.state_dict(), os.path.join(config['output_dir'], config['name'], 'best_val_avg_iou.pth'))
            best_val_avg_iou = val_avg_iou
            print("=> saved best val avg model")
            trigger = 0

        if test_avg_iou > best_test_avg_iou:
            torch.save(model.state_dict(), os.path.join(config['output_dir'], config['name'], 'best_test_avg_iou.pth'))
            best_test_avg_iou = test_avg_iou
            print("=> saved best test avg model")
            trigger = 0


        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), os.path.join(config['output_dir'], config['name'], '{}.pth'.format(str(epoch))))

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
