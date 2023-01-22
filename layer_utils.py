# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:20:49 2021

@author: nkorngut
"""

import torch.nn as nn
from fftc import ifft2c, fft2c



def GenConvBlock(n_conv_layers, in_chan, out_chan, feature_maps):
    conv_block = [nn.Conv2d(in_chan, feature_maps, 3, 1, 1),
                  nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    for _ in range(n_conv_layers - 2):
        conv_block += [nn.Conv2d(feature_maps, feature_maps, 3, 1, 1),
                       nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    return nn.Sequential(*conv_block, nn.Conv2d(feature_maps, out_chan, 3, 1, 1))

def GenUnet():
    return 

def GenFcBlock(feat_list=[512, 1024, 1024, 512]):
    FC_blocks = []
    len_f = len(feat_list)
    for i in range(len_f - 2):
        FC_blocks += [nn.Linear(feat_list[i], feat_list[i + 1]),
                      nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        
    return nn.Sequential(*FC_blocks, nn.Linear(feat_list[len_f - 2], feat_list[len_f - 1]))



def DataConsist(input_, k, m, is_k=False, final=False):
    """
    data consist term, replace the reconstructed data with the known acquired data at places the placed that was sampled.
    :param input_: reconstructed image (image domain)
    :param k: the undersampled k-space
    :param m: mask used to sample the k-space
    :param is_k:
    :param final: indicates if it is the final iteration, if so the function returns also the reconstructed k-space (in addition to the image domain)
    :return:
    """


    if is_k:
      return input_ * m + k * (1 - m)
    else:

      input_p = input_.permute(0,2,3,1); k_p = k.permute(0,2,3,1); m_p = m.permute(0,2,3,1)
      out_kspace = (fft2c(input_p)*(1 - m_p) + m_p*k_p).permute(0,3,1,2)
      out_image = ifft2c(out_kspace.permute(0,2,3,1)).permute(0,3,1,2)

      if final:
            return {'recon_ksapce': out_kspace, 'recon_image': out_image}
      else:
            return out_image


