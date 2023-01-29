#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:56:35 2021

@author: nitzanavidan@bm.technion.ac.il
"""

import matplotlib.pyplot as plt
from fftc import complex_abs

import torch
from metrics import evaluate
from DataLoader_kspace import DataLoader, build_args
from models import KIKI
from models import UNet
from fftc import ifft2c, fft2c

unet_arc = True

models_path = "/tcmldrive/Noam/AIsap/MRI_reconstruction/R[3]_lamb0.8_bs1_channels64_lr5e-05_layers25_Unet/_199_final.pt"

#"/tcmldrive/Noam/AIsap/MRI_reconstruction/R[2]_lamb0.8_bs1_channels64_lr5e-05_layers25_loss_SSIM/_199_final.pt"
#"/tcmldrive/Noam/AIsap/MRI_reconstruction/R[2]_lamb0.8_bs1_channels64_lr5e-05_layers25_loss_MSE/_199_final.pt"
#"/tcmldrive/Noam/AIsap/MRI_reconstruction/checkpoints/Res_R[4]_lamb1_bs1_channels64_lr0.001/_20.pt"
#"/home/nitzanavidan/MRI_Noam_Nitzan/Model2/Res_R[4]_lamb1_bs1_channels64_lr0.001/_20.pt"
args = build_args()
val_loader = DataLoader(args, train=False, val=False, test=True)
print(args)
#model = UNet(args)# KIKI(args)
if (args.model_name == 'KIKI'):
    model = KIKI(args)
elif(args.model_name == 'UNet'):
    model = UNet( in_ch=2,out_ch=2,chans= 2, num_pool_layers = 4, drop_prob = 0.0)

device = torch.device('cpu')
checkpoint = torch.load(models_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
exp_params = checkpoint['exp_params']
train_loss = exp_params['train_losses_plot']
validation_loss = exp_params['validation_losses']

num_images = len(val_loader.dataset)
j=0

img_rec = torch.zeros((num_images,256,256))
img_trg = torch.zeros((num_images,256,256))
img_us = torch.zeros((num_images,256,256))

for i, valid_data in enumerate(val_loader, 0):
    model.eval()
    valid_kspaces_us = valid_data['kspace_us'].to(device, non_blocking=True, dtype=torch.float)
    valid_imgs_fs    = valid_data['img_fs'].to(device, non_blocking=True, dtype=torch.float)
    valid_mask       = valid_data['mask_rev'].to(device, non_blocking=True, dtype=torch.float)
    valid_kspaces_fs = valid_data['kspace_fs'].to(device, non_blocking=True, dtype=torch.float)
    valid_img_us = valid_data['img_us']
    valid_img_fs = valid_data['img_fs']

    if(args.model_name == 'UNet'):
        rec = model(valid_img_us)
        valid_imgs_rec = rec
        print(valid_imgs_rec.shape)
        print(valid_imgs_rec.shape)
        valid_kspace_rec = fft2c(valid_imgs_rec.permute(0,2,3,1))
        valid_kspace_rec = valid_kspace_rec.permute(0,3,1,2)
    elif (args.model_name == 'KIKI'):
        rec = model(valid_kspaces_us, valid_mask)
        valid_imgs_rec = rec['recon_image']
        valid_kspace_rec = rec['recon_ksapce']
    

    # metrics evaluations 
    for k in range(valid_kspaces_us.size()[0]):
        img_trg[j,...] = complex_abs(valid_img_fs[0,...].permute(1,2,0))
        img_rec[j,...] = complex_abs(valid_imgs_rec[0,...].permute(1,2,0))
        img_us[j,...] = complex_abs(valid_img_us[0,...].permute(1,2,0))
        j+= 1

    # plot results 
    if i ==3:
        plt.figure()
        plt.imshow(valid_mask[0,0,:,:], cmap = 'gray'), plt.title('Mask')
        plt.show()
        
        valid_image_rec = torch.squeeze(valid_imgs_rec[0,:,:,:],0)
        valid_kspace_rec = torch.squeeze(valid_kspace_rec[0,:,:,:],0)
        valid_img_fs = torch.squeeze(valid_img_fs[0:,:,:],0)
        valid_img_us = torch.squeeze(valid_img_us[0,:,:,:],0)
        
        valid_image_rec = complex_abs(valid_image_rec.permute(1,2,0)).detach().cpu().numpy()
        valid_kspace_rec    = complex_abs(valid_kspace_rec.permute(1,2,0)).detach().cpu().numpy()
        valid_img_fs    = complex_abs(valid_img_fs.permute(1,2,0)).detach().cpu().numpy()
        valid_img_us    = complex_abs(valid_img_us.permute(1,2,0)).detach().cpu().numpy()
        

        plt.figure()
        plt.subplot(221)
        plt.imshow(valid_img_fs,cmap='gray'),plt.title('Fully sampled')
        plt.subplot(222)
        plt.imshow(valid_image_rec,cmap='gray'),plt.title('Image recon')
        plt.subplot(223)
        plt.imshow(valid_img_us,cmap='gray'),plt.title('Undersampled')
        plt.subplot(224)
        plt.imshow(1-abs(valid_img_fs-valid_image_rec),cmap='gray'),plt.title('Diff fully-recon')
        #plt.show()
        plt.savefig("/tcmldrive/Noam/AIsap/MRI_reconstruction/R[3]_lamb0.8_bs1_channels64_lr5e-05_layers25_Unet/plot.png")
    
metrics_rec = evaluate(img_trg.detach().cpu().numpy(), img_rec.detach().cpu().numpy())
print(f'metrics_rec={metrics_rec}')

