#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 12:21:20 2021

@author: nitzanavidan@bm.technion.ac.il
"""
from argparse import ArgumentParser
import torch
from pathlib import Path
import torch.nn as nn
import os
#import utils
from importlib import import_module

from mri_data_kspace import fetch_dir
from subsample import create_mask_for_mask_type
from transforms_kspace import KIKIDataTransform
from mri_data_kspace import SliceDataset


def build_args(epochs=1):
    parser = ArgumentParser()

    # basic args
    path_config = Path("fastmri_dirs.yaml")
    data_path = fetch_dir("knee_path", path_config) / 'singlecoil_train'
    val_path = fetch_dir("knee_path", path_config) / 'singlecoil_val'
    test_path = fetch_dir("knee_path", path_config) / 'singlecoil_val'
    batch_size =   1

    # set defaults based on optional directory config
    default_root_dir = fetch_dir("log_path", path_config) / "unet_train" 

    # client arguments
    parser.add_argument("--mode", default="train", choices=("train", "test"), type=str,help="Operation mode" )

    # data transform params
    parser.add_argument( "--mask_type", choices=("random", "equispaced"), default="equispaced", type=str, help="Type of k-space mask" )
    parser.add_argument( "--center_fractions",  nargs="+",  default=[0.16], type=float, help="Number of center lines to use in mask" )
    parser.add_argument("--accelerations", nargs="+", default=[2], type=int,  help="Acceleration rates to use for masks" )
    
    parser.add_argument( "--num_workers", default=8, type=float, help="Number of workers to use in data loader")
    
    parser.add_argument( "--data_path", default=None, type=Path, help="Path to fastMRI data root")
    
    parser.add_argument( "--val_path", default=None, type=Path, help="Path to fastMRI val data root")
    
    parser.add_argument( "--test_path", default=None, type=Path, help="Path to data for test mode. This overwrites data_path and test_split")
    
    parser.add_argument( "--default_root_dir", default=None, type=Path, help="directory for logs and checkpoints")
        
    parser.add_argument("--challenge", choices=("singlecoil", "multicoil"), default="singlecoil", type=str, help="Which challenge to preprocess for")
    
    
    parser.add_argument("--sample_rate", default=None, type=float,
            help="Fraction of slices in the dataset to use (train split only). If not given all will be used. Cannot set together with volume_sample_rate.")
    
    parser.add_argument("--volume_sample_rate",  default=None, type=float,
            help="Fraction of volumes of the dataset to use (train split only). If not given all will be used. Cannot set together with sample_rate.")
    
    parser.add_argument("--loss",default=nn.L1Loss(),type=float, help="loss function")

    
    parser.add_argument('--add_path_str', type=str, default='', help='string to be added to directory name')
    parser.add_argument('--model_name', type=str, default='KIKI', help='model name')
    parser.add_argument('--dataset_name', type=str, default='HCP_MGH_T1w', help='name of the dataset')
    parser.add_argument('--acc_rate', type=float, default=5.02, help='acceleration ratio')
    parser.add_argument('--acs_num', type=int, default=16, help='the number of acs lines')
    parser.add_argument('--mask_name', type=str, default='', help='mask name')
    
    parser.add_argument('--gpu_alloc', type=int, default=3, help='GPU allocation (0 to 3)')
    
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.00005, help='adam: learning rate') # start: 0.00005
    parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--lambda_', type=float, default=0.8, help='lamba:loss regularizeation term')
    
    parser.add_argument('--im_height', type=int, default=256, help='size of image height')
    parser.add_argument('--im_width', type=int, default=256, help='size of image width')
    parser.add_argument('--channels', type=int, default=2, help='number of image channels')
    
    parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--sample_interval_valid', type=int, default=200, help='interval between sampling of images from generators')
    parser.add_argument('--sample_interval_train', type=int, default=400, help='interval between sampling of images from generators')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between model checkpoints')
    parser.add_argument('--mask_index', type=int, default=0, help='validation/test mask index')
    
    parser.add_argument('--data_augment', type=bool, default=False, help='32-fold data augmentation')
    parser.add_argument('--random_sampling', type=bool, default=False, help='Generate random sampling patterns during training')
    
    parser.add_argument('--k', type=int, default=25)
    parser.add_argument('--i', type=int, default=25)
    parser.add_argument('--iters', type=int, default=2)    
    parser.add_argument('--fm', type=int, default=64)

    args = parser.parse_args()
    
    parser.add_argument('--in_ch',  type=int, default=args.channels)
    parser.add_argument('--out_ch', type=int, default=args.channels)
    

    
    # data config with path to fastMRI data and batch size
    parser.set_defaults(data_path=data_path, val_path=val_path, test_path=test_path,
                        batch_size=batch_size, default_root_dir=default_root_dir,
                        epochs = epochs
                        )

    
    args = parser.parse_args()
    
    return args

def DataLoader(args, train=True, val=False, test=False):
    """
    Args:
        args: args informationof data from function build_args()
        train: bool, for definig path
        val: bool, for definig path
        test: bool, for definig path
    Returns:
        dataloader.
    """
    if train:
        data_path = args.data_path
    elif val:
        data_path = args.val_path
    elif test:
        data_path = args.test_path
    
    # data processing 
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )

    # use random masks for train transform, fixed masks for val transform
    
    transform = KIKIDataTransform(args.challenge, mask_func=mask, use_seed=False)
  
    dataset= SliceDataset(
        root=data_path,
        transform=transform,
        challenge=args.challenge,
        train = train,
        test = test,
        val = val
    )

  
    dataloader= torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=True
        )
    
    return dataloader


'''
if __name__ == "__main__":
    opt = build_args()

    #opt = GenConfig()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]='%d' % opt.gpu_alloc

    add_path_str = utils.GenAddPathStr(opt)
    os.makedirs('SavedModels_%s/%s' % (opt.model_name, add_path_str), exist_ok=True)
    os.makedirs('Validation_%s/%s' % (opt.model_name, add_path_str), exist_ok=True)

    GeneratorNet = getattr(import_module('models'), opt.model_name)
    # Data loader
    dataloader_train = DataLoader(opt, train=True, val=False)
    dataloader_valid = DataLoader(opt, train=False, val=True)
    
    generator = GeneratorNet(opt)
    # train_loader = DataLoader(args, train=True, val=False)
    # val_loader = DataLoader(args, train=False, val=True)
    # num=0
    # for i, batch in enumerate(train_loader,0):
        
    #     inputs = batch[0]
    #     outputs = batch[1]
    #     num+=batch[0].size()[0]
'''