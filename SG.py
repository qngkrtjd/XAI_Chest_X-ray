import argparse
import os
import sys

import numpy as np
from scipy import misc
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import densenet121
from torchvision.utils import save_image

from lib.gradients import SmoothGrad
from lib.image_utils import preprocess_image, save_as_gray_image, show_cam_on_image
from lib.label import NIH_LABELS

import models_vit,models_densenet
from torchsummary import summary
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--img', type=str, default='./00014687_001.png',
                        help='Input image path')
    parser.add_argument('--out_dir', type=str, default='./result/grad/',
                        help='Result directory path')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Sample size of SmoothGrad')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")
    if args.img:
        print('Input image: {}'.format(args.img))
    else:
        print('Input image: raccoon face (scipy.misc.face())')
    print('Output directory: {}'.format(args.out_dir))
    print('Sample size of SmoothGrad: {}'.format(args.n_samples))
    print()
    return args

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--train_list", default=None, type=str, help="file for train list")
    parser.add_argument("--val_list", default=None, type=str, help="file for val list")
    parser.add_argument("--test_list", default=None, type=str, help="file for test list")
    parser.add_argument('--eval_interval', default=10, type=int)
    parser.add_argument('--fixed_lr', action='store_true', default=False)
    parser.add_argument('--vit_dropout_rate', type=float, default=0,
                        help='Dropout rate for ViT blocks (default: 0.0)')
    parser.add_argument("--build_timm_transform", action='store_true', default=False)
    parser.add_argument("--aug_strategy", default='default', type=str, help="strategy for data augmentation")
    parser.add_argument("--dataset", default='chestxray', type=str)

    parser.add_argument('--repeated-aug', action='store_true', default=False)

    parser.add_argument("--optimizer", default='adamw', type=str)

    parser.add_argument('--ThreeAugment', action='store_true')  # 3augment

    parser.add_argument('--src', action='store_true')  # simple random crop

    parser.add_argument('--loss_func', default=None, type=str)

    parser.add_argument("--norm_stats", default=None, type=str)

    parser.add_argument("--checkpoint_type", default=None, type=str)

    return parser

def main():
    args = parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    target_layer_names = ['35']
    target_index = None

    is_vit=True

    # Prepare input image
    img = cv2.imread(args.img, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    preprocessed_img = preprocess_image(img, args.cuda)

    #model    
    
    if(is_vit):
        model=models_vit.vit_small_patch16()
    else:
        model=torchvision.models.densenet121(num_classes=14)
        #model=models_densenet.DenseNet121()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if(is_vit):
        _state_dict=torch.load("./finetuned_models/vit-s_CXR_0.3M_mae.pth",map_location=device)  
    else:
        _state_dict=torch.load("./finetuned_models/densenet121_CXR_0.3M_mae.pth",map_location=device)  
    
    a=_state_dict.get('model')

    del _state_dict['optimizer']
    del _state_dict['epoch']
    del _state_dict['scaler']
    del _state_dict['args']
    b = { k.replace('fea','model.fea'): v for k, v in a.items() }
    
    c = { k.replace('classifier','model.classifier.0'): v for k, v in b.items() }
    d = { k.replace('fc_norm','norm'): v for k, v in a.items() }

    if(is_vit):
        model.load_state_dict(d)
    else:
        model.load_state_dict(a)
    
    model.eval()

    # Prediction
    output = model(preprocessed_img)
    pred_index = np.argmax(output.detach().cpu().numpy())
    print(output)
    print(pred_index)
    print('Prediction: {}'.format(NIH_LABELS[pred_index]))
    
    # Compute smooth gradient
    smooth_grad = SmoothGrad(
        pretrained_model=model,
        cuda=args.cuda,
        n_samples=args.n_samples,
        magnitude=True)
    smooth_saliency = smooth_grad(preprocessed_img, index=target_index)
    
    if(is_vit):
        save_as_gray_image(smooth_saliency, os.path.join(args.out_dir, 'smooth_grad_vit.jpg'))    
    else:
        save_as_gray_image(smooth_saliency, os.path.join(args.out_dir, 'smooth_grad_densenet.jpg'))
    print('Saved smooth gradient image')

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    visualization = show_cam_on_image(img, smooth_saliency, use_rgb=True)
    plt.imshow(visualization)
    plt.show()
    

if __name__ == '__main__':
    main()
