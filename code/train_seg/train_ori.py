import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler as ampgs
from torch.cuda.amp import autocast as autocast

import yaml
from albumentations.augmentations import transforms, geometric
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter


import segmentation_models_pytorch as smp
import archs
import losses
from dataset import Dataset
from metrics import iou_score, iou_score_per_class
from utils import AverageMeter, str2bool
from radam import RAdam

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


ARCH_NAMES = ARCH_DICT.keys()
LOSS_NAMES = list(LOSS_DICT.keys())
LOSS_NAMES.append('BCEWithLogitsLoss')
MODE_NAMES = ['binary', 'multiclass', 'multilabel']
ACT_NAMES = [None, 'identity',  'sigmoid', 'softmax2d', 'softmax', 'logsoftmax', 'tanh', 'argmax', 'argmax2d']

import albumentations as albu
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')
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
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
    else:
         _transform = [
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
    return albu.Compose(_transform)


def scan_files(input_file_path, ext_list):
    file_list = []
    for root, dirs, files in os.walk(input_file_path):
        # scan all files and put it into list

        for f in files:
            if os.path.splitext(f)[1].lower() in ext_list:
                file_list.append(os.path.join(root, f).replace("\\","/").replace(os.path.join(input_file_path, "").replace("\\","/"), "", 1 ))

    return file_list


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
    parser.add_argument('--input_w', default=512, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=512, type=int,
                        help='image height')

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
    parser.add_argument('--dataset_root', default='input',
                        help='dataset root path')
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.pkl',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD', 'RAdam'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD', 'RAdam']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dp_rate', default=-1., type=float)
    parser.add_argument('--class_name', default='nlst,cn',
                        help='class name list')

    parser.add_argument('--tf_log_path', default=None, type=str)


    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):


    per_cls_avg_meter = [AverageMeter() for _ in range(config['num_classes'])]
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:

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
                    iou, iou_per_cls = iou_score_per_class(outputs[-1], target)
                else:
                    output = model(input)

                    loss = criterion(output, target)
                    iou, iou_per_cls = iou_score_per_class(output, target)
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
                iou, iou_per_cls = iou_score_per_class(outputs[-1], target)
            else:
                output = model(input)

                loss = criterion(output, target)
                iou, iou_per_cls = iou_score_per_class(output, target)

            # compute gradient and do optimizing step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        for tmp_m, tmp_v in zip(per_cls_avg_meter, iou_per_cls):
            tmp_m.update(tmp_v, input.size(0))

        possfix_list = [('loss', avg_meters['loss'].avg),('iou', avg_meters['iou'].avg),]

        for i, cls_name in enumerate(config['class_name']):
            possfix_list.append(('iou_{}'.format(cls_name), per_cls_avg_meter[i].avg))


        postfix = OrderedDict(possfix_list)
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    possfix_list = [('loss', avg_meters['loss'].avg),('iou', avg_meters['iou'].avg),]
    for i, cls_name in enumerate(config['class_name']):
        possfix_list.append(('iou_{}'.format(cls_name), per_cls_avg_meter[i].avg))

    return OrderedDict(possfix_list)


def validate(config, val_loader, model, criterion):
    per_cls_avg_meter = [AverageMeter() for _ in range(config['num_classes'])]
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, iou_per_cls = iou_score_per_class(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, iou_per_cls = iou_score_per_class(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            for tmp_m, tmp_v in zip(per_cls_avg_meter, iou_per_cls):
                tmp_m.update(tmp_v, input.size(0))

            possfix_list = [('loss', avg_meters['loss'].avg),('iou', avg_meters['iou'].avg),]

            for i, cls_name in enumerate(config['class_name']):
                possfix_list.append(('iou_{}'.format(cls_name), per_cls_avg_meter[i].avg))


            postfix = OrderedDict(possfix_list)

            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    possfix_list = [('loss', avg_meters['loss'].avg),('iou', avg_meters['iou'].avg),]
    for i, cls_name in enumerate(config['class_name']):
        possfix_list.append(('iou_{}'.format(cls_name), per_cls_avg_meter[i].avg))

    return OrderedDict(possfix_list)


def main():
    config = vars(parse_args())

    if config['act'] == None:
        if config['mode'] == 'multiclass':
            config['act'] = 'softmax2d' # could be None for logits or 'softmax2d' for multicalss segmentation
        else:
            config['act'] = None

    config['class_name'] = config['class_name'].rstrip().split(',')
    if len(config['class_name']) != config['num_classes']:
        raise Exception('num of class name not equal to num_classes in config file.')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    tf_log_path = config['tf_log_path']
    tf_save_name = '{}_{}_{}_{}_{}'.format(
                                           config['name'], config['arch'], config['loss'],
                                           config['optimizer'], str(config['lr'])
                                          )

    writer = None
    if tf_log_path is not None:
        writer = SummaryWriter(os.path.join(tf_log_path, tf_save_name),comment=tf_save_name, flush_secs=1)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)


    if writer is not None:
        config_content = ''
        for k in sorted(config.keys()):
            config_content += '{}: {}\n\n'.format(k, str(config[k]))
        writer.add_text('config:', config_content)


    # define loss function (criterion)
    config['loss'] = config['loss'].rstrip().split(',')
    config['loss_weight']= config['loss_weight'].rstrip().split(',')

    assert len(config['loss']) == len(config['loss_weight']), \
           'num of loss must equal to num of loss weight, {} vs {}'.format(str(len(config['loss'])), str(len(config['loss_weight'])))

    loss_list = []
    loss_w_list = []
    for loss_name, loss_w in zip(config['loss'], config['loss_weight']):
        if loss_name == 'BCEWithLogitsLoss':
            #criterion = smp.utils.losses.DiceLoss()
            current_criterion = nn.BCEWithLogitsLoss()
        else:
            current_criterion = LOSS_DICT[loss_name](mode=config['mode'])
        loss_list.append(current_criterion)
        loss_w_list.append(float(loss_w))

    criterion = JointLoss(loss_list, loss_w_list)

    # if len(config['loss']) == 1:
    #     loss_name = config['loss'][0]
    #     if config['loss'] == 'BCEWithLogitsLoss':
    #         #criterion = smp.utils.losses.DiceLoss()
    #         criterion = nn.BCEWithLogitsLoss()
    #     else:
    #         criterion = LOSS_DICT[config['loss']](mode=config['mode'])

    criterion = criterion.cuda()
    cudnn.benchmark = True

####################################################################################
    # create model


    # ENCODER = 'timm-resnest14d'
    # ENCODER_WEIGHTS = 'imagenet'


    # create segmentation model with pretrained encoder
    model = ARCH_DICT[config['arch']](
        encoder_name=config['encoder'],
        encoder_weights=config['encoder_weight'],
        classes=config['num_classes'],
        activation=config['act'],
    )

    if config['encoder_weight'] is not None:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(config['encoder'], config['encoder_weight'])
    else:
        preprocessing_fn = None

    # print("=> creating model %s" % config['arch'])
    # model = archs.__dict__[config['arch']](config['num_classes'],
    #                                        config['input_channels'],
    #                                        config['deep_supervision'],
    #                                        config['dp_rate'])




##############################################################################
    if torch.cuda.device_count() > 1:
        print('number of GPU > 1, using data parallel')
        model = nn.DataParallel(model)

    model = model.cuda()
    if config['show_arch']:
        from torchsummary import summary
        summary(model, input_size=(config['input_channels'], config['input_h'], config['input_w']))

    '''
    for name, param in model.named_parameters():
        print(name,param.requires_grad)
        param.requires_grad=False
    '''


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
    else:
        raise NotImplementedError

    # Data loading code
    img_ids = scan_files(os.path.join(config['dataset_root'], config['dataset'], 'images'), [config['img_ext']])
    # img_ids = glob(os.path.join(config['dataset_root'], config['dataset'], 'images', '*' + config['img_ext']), recursive=True)
    # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    # img_ids = [os.path.splitext(p)[0].replace(os.path.join()) for p in img_ids]
    img_ids = [os.path.splitext(p)[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    train_transform = Compose([
        geometric.rotate.RandomRotate90(),
        transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),
        geometric.resize.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        geometric.resize.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['dataset_root'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['dataset_root'], config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        preprocessing=get_preprocessing(preprocessing_fn),
        transform=train_transform,
        mode=config['mode'])
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['dataset_root'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['dataset_root'], config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        preprocessing=get_preprocessing(preprocessing_fn),
        transform=val_transform,
        mode=config['mode'])

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

    log_list = [
                ('epoch', []),
                ('lr', []),
                ('loss', []),
                ('iou', []),
                ('val_loss', []),
                ('val_iou', []),
               ]

    for i, cls_name in enumerate(config['class_name']):
        log_list.append(('iou_{}'.format(cls_name), []))
    for i, cls_name in enumerate(config['class_name']):
        log_list.append(('val_iou_{}'.format(cls_name), []))

    log = OrderedDict(log_list)

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        train_msg = ''
        val_msg = ''
        for i, cls_name in enumerate(config['class_name']):
            train_msg += 'train_iou_{} {:04f} - '.format(cls_name, train_log['iou_{}'.format(cls_name)])
        train_msg = train_msg[:-2]
        for i, cls_name in enumerate(config['class_name']):
            val_msg += 'val_iou_{} {:04f} - '.format(cls_name, val_log['iou_{}'.format(cls_name)])
        val_msg = val_msg[:-2]
        print(train_msg)
        print(val_msg)


        if writer is not None:
            writer.add_scalar('Train_Loss', train_log['loss'], global_step=epoch)
            writer.add_scalar('Val_Loss', val_log['loss'], global_step=epoch)
            writer.add_scalar('Train_IoU', train_log['iou'], global_step=epoch)
            writer.add_scalar('Val_IoU', val_log['iou'], global_step=epoch)
            for i, cls_name in enumerate(config['class_name']):
                writer.add_scalar('Train_{}_IoU'.format(cls_name), train_log['iou_{}'.format(cls_name)], global_step=epoch)
            for i, cls_name in enumerate(config['class_name']):
                writer.add_scalar('Val_{}_IoU'.format(cls_name), val_log['iou_{}'.format(cls_name)], global_step=epoch)

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        for i, cls_name in enumerate(config['class_name']):
            log['iou_{}'.format(cls_name)].append(train_log['iou_{}'.format(cls_name)])

        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        for i, cls_name in enumerate(config['class_name']):
            log['val_iou_{}'.format(cls_name)].append(val_log['iou_{}'.format(cls_name)])


        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
