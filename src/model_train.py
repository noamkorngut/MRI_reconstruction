#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 18:17:14 2021

@author: noam korngut
"""

import numpy as np
import torch
from datetime import datetime
import random
import tqdm
import os
import torch.nn as nn
#import utils
#import loss_functions
#from loss_functions import MS_SSIM_L1_LOSS, ssim
import loss_functions
from DataLoader_kspace import DataLoader, build_args
from models import KIKI, KIKI_res
from models import UNet
from fftc import ifft2c, fft2c

from loss_functions import SSIMLoss
#from torchmetrics.functional import structural_similarity_index_measure as SSIM
#from fastmri.pl_modules import FastMriDataModule, UnetModule
#from pytorch_ssim import SSIM


def get_exp_name ():

    now = datetime.now() # current date and time
    exp_name = args.model_name + '_models' + now.strftime("%Y_%m_%d_%H_%M_%S")
    
    return exp_name



def train(args, load_model= False, model_path=None):
    
    out_folder = f'./R{args.accelerations}_lamb{args.lambda_}_bs{args.batch_size}_channels{args.fm}_lr{args.lr}_layers{args.k}_loss_L1/'
    
    try:
        os.mkdir (out_folder)
    except:
        pass

    save_each = 10
    report_each = args.batch_size*10
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    pre_epoch=0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
    # load the data- gets a dictionary 
    train_loader = DataLoader(args, train=True, val=False)
    val_loader = DataLoader(args, train=False, val=True)  

    # defining the model, optimizer and loss 
    if (args.model_name == 'KIKI'):
        model = KIKI(args)
        #model = KIKI_res(args)
    elif (args.model_name == 'UNet'):
        model = UNet(
            in_ch = args.in_ch,
            out_ch = args.out_ch,
            chans = args.channels, #args.chans,
            num_pool_layers = 4, #args.num_pool_layers,
            drop_prob = 0.0, #args.drop_prob,
            #lr=args.lr,
            #lr_step_size=40, #args.lr_step_size,
            #lr_gamma=0.1,#args.lr_gamma,
            #weight_decay=0.0,#args.weight_decay,
        )


    model = model.to(device, non_blocking=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=10**-7)

    if(args.loss == 'MSE'):
        criterion = torch.nn.MSELoss()
    elif(args.loss == 'L1'):
        criterion = nn.L1Loss() #SSIMLoss() #SSIM
    #elif(args.loss == 'SSIM'):
    #    criterion = loss_functions.SSIMLoss


    if load_model:
        checkpoint = torch.load(model_path)
        exp_params = checkpoint['exp_params']
        pre_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    train_losses = []
    train_losses_plot = []          
    validation_losses = []
    
    print(f'batch_size={args.batch_size},lambda_={args.lambda_},R={args.accelerations},fm={args.fm}')
    for epoch in range(pre_epoch, args.epochs):

        # training part
        model.train()
       
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Train: Epoch {}, lr {}'.format(epoch, args.lr))
        
        epoch_trian_loss = 0
        total_validaiton_loss = 0
        mean_train_loss = 0
        

        for i, train_data in enumerate(train_loader,0):
            train_kspaces_us  = train_data['kspace_us'].to(device, non_blocking=True, dtype=torch.float)
            train_imgs_fs     = train_data['img_fs'].to(device, non_blocking=True, dtype=torch.float)
            train_mask        = train_data['mask_rev'].to(device, non_blocking=True, dtype=torch.float)
            train_kspaces_fs  = train_data['kspace_fs'].to(device, non_blocking=True, dtype=torch.float)

            if (args.model_name == 'KIKI'):
                rec =  model(train_kspaces_us, train_mask)
                train_imgs_rec = rec['recon_image']
                train_kspace_rec = rec['recon_ksapce']
            elif (args.model_name == 'UNet'):
                #print(train_mask.shape, 'mask')

                # convert to image space
                #magnitude_img = torch.sqrt(torch.sum(torch.pow(train_kspaces_us.cpu(),2), axis=1))
                print(train_kspaces_us.shape, 'before ifft')
                train_img_us= ifft2c(train_kspaces_us.permute(0,2,3,1))
                print(train_img_us.shape, 'after ifft')
                #mag2 = torch.norm(train_kspaces_us.squeeze(0),dim=0)
                train_img_us.to(device, non_blocking=True, dtype=torch.float)
                #print(magnitude_img.shape, 'magnitude #################')
                rec =  model(train_img_us.permute(0,3,1,2)) #model(train_kspaces_us) #model(train_kspaces_us, train_mask) .permute(0,2,3,1)
                train_imgs_rec = rec
                if train_imgs_rec.shape[-1] == 2:
                    train_imgs_rec = train_imgs_rec.permute(0,3,1,2)
                train_kspace_rec = fft2c(train_imgs_rec.permute(0,2,3,1)).permute(0,3,1,2)

                #loss_train = criterion(torch.norm(train_imgs_rec.squeeze(0), dim=0), torch.norm(train_imgs_fs.squeeze(0), dim=0), arr[np.newaxis, :])# + args.lambda_* criterion(train_kspace_rec*(1-train_mask), train_kspaces_fs*(1-train_mask),torch.max(train_kspaces_fs))

            loss_train = (criterion((train_imgs_rec), (train_imgs_fs)))  + args.lambda_* (criterion(train_kspace_rec*(1-train_mask), train_kspaces_fs*(1-train_mask)))
            #print(loss_train)

            epoch_trian_loss += loss_train.item()
            train_losses.append(loss_train.item())
                 

            # zero the parameter gradients
            optimizer.zero_grad()
            batch_size = train_data['kspace_us'].size()[0]
            loss_train.backward()
            optimizer.step()
            
            # step += 1
            tq.update(batch_size)
            
            mean_train_loss = np.mean(train_losses[-report_each:])
            tq.set_postfix(loss='{:.5f}'.format(mean_train_loss))
        
        
        
        train_losses_plot.append(mean_train_loss)
        tq.close()
            
        with torch.no_grad():
            
            model.eval()
            
            epoch_val_loss = 0
            
            tq = tqdm.tqdm(total=(len(val_loader) * batch_size))
            tq.set_description('Validation: Epoch {}, lr {}'.format(epoch, args.lr))
            
            for i, valid_data in enumerate(val_loader):

                valid_kspaces_us = valid_data['kspace_us'].to(device, non_blocking=True, dtype=torch.float)
                valid_imgs_fs    = valid_data['img_fs'].to(device, non_blocking=True, dtype=torch.float)
                valid_mask       = valid_data['mask_rev'].to(device, non_blocking=True, dtype=torch.float)
                valid_kspaces_fs  = valid_data['kspace_fs'].to(device, non_blocking=True, dtype=torch.float)
                if (args.model_name == 'KIKI'):
                    rec = model(valid_kspaces_us, valid_mask)
                    valid_imgs_rec = rec['recon_image']
                    valid_kspace_rec = rec['recon_ksapce']
                elif(args.model_name == 'UNet'):
                    valid_img_us = ifft2c(valid_kspaces_us.permute(0,2,3,1))
                    valid_img_us.to(device, non_blocking=True, dtype=torch.float)
                    rec = model(valid_img_us.permute(0,3,1,2))
                    valid_imgs_rec = rec
                #loss_valid = criterion(valid_imgs_rec, valid_imgs_fs, torch.max(valid_imgs_fs)) #+ args.lambda_* criterion(valid_kspace_rec*(1-valid_mask), valid_kspaces_fs*(1-valid_mask), torch.max(valid_kspaces_fs))
                loss_valid = (criterion(valid_imgs_rec, valid_imgs_fs)) + args.lambda_* (criterion(valid_kspace_rec*(1-valid_mask), valid_kspaces_fs*(1-valid_mask)))


                epoch_val_loss += loss_valid.item()
               
                tq.update(batch_size)
                tq.set_postfix(loss='{:.5f}'.format(epoch_val_loss/(i+1)))
                

            epoch_val_loss = epoch_val_loss/len(val_loader)
            validation_losses.append(epoch_val_loss)  
            tq.close()
        
            
        if epoch > 0  and epoch % save_each == 0:
            
            exp_params = {}
            exp_params ['validation_losses'] = validation_losses
            exp_params ['train_losses_plot'] = train_losses_plot
           
            exp_name = out_folder 
            model_path = f"{exp_name}_{epoch}.pt"
            
            if torch.cuda.device_count() == 1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': args.loss,
                    'exp_params': exp_params}, model_path)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(), #model.module.state_dict()
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': args.loss,
                    'exp_params': exp_params}, model_path)
                

    exp_params = {}    
    exp_params ['validation_losses'] = validation_losses
    exp_params ['train_losses_plot'] = train_losses_plot
   
    exp_name = out_folder
    model_path = f"{exp_name}_{epoch}_final.pt"
    if torch.cuda.device_count() == 1:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': args.loss,
            'exp_params': exp_params}, model_path)
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(), #model.module.state_dict()
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': args.loss,
            'exp_params': exp_params}, model_path)

    torch.cuda.empty_cache()
    

if __name__=="__main__":
    args = build_args(epochs=200)
    train(args)