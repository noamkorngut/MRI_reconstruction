# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:19:05 2021

@author: nkorngut
"""

import torch
import torch.nn as nn
from layer_utils import GenConvBlock, DataConsist
from fftc import ifft2c, fft2c
from torch.nn import functional as F

class KIKI(nn.Module):
    def __init__(self, m):
        super(KIKI, self).__init__()

        conv_blocks_K = [] 
        conv_blocks_I = []
        
        for i in range(m.iters):
            conv_blocks_K.append(GenConvBlock(m.k, m.in_ch, m.out_ch, m.fm))
            conv_blocks_I.append(GenConvBlock(m.i, m.in_ch, m.out_ch, m.fm))

        self.conv_blocks_K = nn.ModuleList(conv_blocks_K)
        self.conv_blocks_I = nn.ModuleList(conv_blocks_I)
        self.n_iter = m.iters
        self.lambda_ = m.lambda_
    def forward(self, kspace_us, mask):        
        rec = kspace_us 
        ### rec in k-space###

        for i in range(self.n_iter):
            rec = self.conv_blocks_K[i](rec)
            rec = ifft2c(rec.permute(0,2,3,1))
            ### rec in image domain
            rec = rec.permute(0,3,1,2)

            rec = rec + self.conv_blocks_I[i](rec)
            if i==self.n_iter - 1:
                rec = DataConsist(rec, kspace_us, mask, final=True)
            
            else:
                rec = DataConsist(rec, kspace_us, mask, final=False)


            if i < self.n_iter - 1:
                rec = fft2c(rec.permute(0,2,3,1))
                rec = rec.permute(0,3,1,2)

        return rec


class KIKI_res(nn.Module):
    def __init__(self, m):
        super(KIKI, self).__init__()

        conv_blocks_K = []
        conv_blocks_I = []

        for i in range(m.iters):
            conv_blocks_K.append(GenConvBlock(m.k, m.in_ch, m.out_ch, m.fm))
            conv_blocks_I.append(GenConvBlock(m.i, m.in_ch, m.out_ch, m.fm))

        self.conv_blocks_K = nn.ModuleList(conv_blocks_K)
        self.conv_blocks_I = nn.ModuleList(conv_blocks_I)
        self.n_iter = m.iters
        self.lambda_ = m.lambda_

    def forward(self, kspace_us, mask):
        rec = kspace_us

        for i in range(self.n_iter):
            rec = rec + self.conv_blocks_K[i](rec)
            rec = ifft2c(rec.permute(0, 2, 3, 1))
            ###rec = image###
            rec = rec.permute(0, 3, 1, 2)

            rec = rec + self.conv_blocks_I[i](rec)
            if i == self.n_iter - 1:
                rec = DataConsist(rec, kspace_us, mask, self.lambda_, final=True) #DataConsist2

            else:
                rec = DataConsist(rec, kspace_us, mask, self.lambda_, final=False) #DataConsist2

            if i < self.n_iter - 1:
                rec = fft2c(rec.permute(0, 2, 3, 1))
                rec = rec.permute(0, 3, 1, 2)

        return rec

### add for AISAP task

    # ------------
    # model
    # ------------


class UNet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234ג€“241.
    Springer, 2015.
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            chans: int = 32,
            num_pool_layers: int = 4,
            drop_prob: float = 0.0,
    ):
        """
        Args:
            in_ch: Number of chnels in the input to the U-Net model.
            out_ch: Number of chnels in the output to the U-Net model.
            ch: Number of output chnels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ch = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_ch, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_ch, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_ch, H, W)`.

        Returns:
            Output tensor of shape `(N, out_ch, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_ch: int, out_ch: int, drop_prob: float):
        """
        Args:
            in_ch: Number of chnels in the input.
            out_ch: Number of chnels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_ch, H, W)`.

        Returns:
            Output tensor of shape `(N, out_ch, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_ch: int, out_ch: int):
        """
        Args:
            in_ch: Number of chnels in the input.
            out_ch: Number of chnels in the output.
        """
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_ch, H, W)`.

        Returns:
            Output tensor of shape `(N, out_ch, H*2, W*2)`.
        """
        return self.layers(image)

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import common



class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

    def bis(self, input, dim, index):
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, V, K, Q):

        ### search
        Q_unfold=F.unfold(Q, kernel_size=(3, 3), padding=1)
        K_unfold=F.unfold(K,kernel_size=(3,3),padding=1)
        K_unfold = K_unfold.permute(0, 2, 1)

        K_unfold = F.normalize(K_unfold, dim=2)  # [N, Hr*Wr, C*k*k]
        Q_unfold= F.normalize(Q_unfold, dim=1)  # [N, C*k*k, H*W]

        R_lv3 = torch.bmm(K_unfold , Q_unfold)  # [N, Hr*Wr, H*W]
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)  # [N, H*W]

        ### transfer
        V_unfold = F.unfold(V, kernel_size=(3, 3), padding=1)

        T_lv3_unfold = self.bis(V_unfold, 2, R_lv3_star_arg)


        T_lv3 = F.fold(T_lv3_unfold, output_size=Q.size()[-2:], kernel_size=(3, 3), padding=1) / (3. * 3.)

        S = R_lv3_star.view(R_lv3_star.size(0), 1, Q.size(2), Q.size(3))

        return S,T_lv3

#transformer

class T2Net(nn.Module):
    def __init__(self, upscale_factor, input_channels, target_channels, n_resblocks, n_feats, res_scale, bn=False, act=nn.ReLU(True), conv=common.default_conv, head_patch_extraction_size=5, kernel_size=3, early_upsampling=False):

        super(T2Net,self).__init__()

        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.scale = res_scale
        self.act = act
        self.bn = bn
        self.input_channels = input_channels
        self.target_channels = target_channels

        m_head1 = [conv(input_channels, n_feats, head_patch_extraction_size)]
        m_head2 = [conv(input_channels, n_feats, head_patch_extraction_size)]

        m_body1 = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale, bn=self.bn
            ) for _ in range(n_resblocks)
        ]
        m_body1.append(conv(n_feats, n_feats, kernel_size))

        m_body2 = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale, bn=self.bn
            ) for _ in range(n_resblocks)
        ]
        m_body2.append(conv(n_feats, n_feats, kernel_size))

        m_conv1=[nn.Conv2d(n_feats*2,n_feats,kernel_size=1) for _ in range(n_resblocks)]

        #head
        self.head1 = nn.Sequential(*m_head1)
        self.head2 = nn.Sequential(*m_head2)

        #body
        self.body1=nn.Sequential(*m_body1)
        self.body2=nn.Sequential(*m_body2)

        #kersize=1 conv
        self.conv1=nn.Sequential(*m_conv1)

        #tail
        m_tail_late_upsampling = [
            common.Upsampler(conv, upscale_factor, n_feats, act=False),
            conv(n_feats, target_channels, kernel_size)
        ]
        m_tail_early_upsampling = [
            conv(n_feats, target_channels, kernel_size)
        ]
        if early_upsampling:
            self.tail = nn.Sequential(*m_tail_early_upsampling)
        else:
            self.tail = nn.Sequential(*m_tail_late_upsampling)#走这个

        self.b_tail=nn.Conv2d(n_feats,target_channels,kernel_size=1)

        #transformer modules
        m_transformers=[Transformer() for _ in range(n_resblocks)]

        self.transformers=nn.Sequential(*m_transformers)

    def forward(self, input):

        x1=self.head1(input)
        x2=self.head2(input)

        res1=x1
        res2=x2

        for i in range(self.n_resblocks):
            x1=self.body1[i](x1)
            x2=self.body2[i](x2)
            S,T=self.transformers[i](x2,x2,x1)
            T=torch.cat([x1,T],1)
            T=self.conv1[i](T)
            x1=x1+T*S

        y1=self.tail(x1+res1)
        y2=self.b_tail(x2+res2)

        return y1,y2


# class UNet(nn.Module):
#     def __init__(self, args):
#         super(UNet, self).__init__()
# 
#         self.conv_block1 = nn.Sequential(
#             nn.Conv2d(args.in_ch, 16, 3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(16, 16, 3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
# 
#         self.conv_block2 = nn.Sequential(
#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
# 
#         self.conv_block3 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
# 
#         self.conv_block4 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
# 
#         self.up_block1 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
# 
#         self.up_block2 = nn.Sequential(
#             nn.Conv2d(64, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU()
#         )
# 
#         self.up_block3 = nn.Sequential(
#             nn.Conv2d(32, 16, 3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(16, 16, 3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU()
#         )
# 
#         self.final_block = nn.Sequential(
#             nn.Conv2d(16, args.out_ch, 1),
#             nn.Tanh()
#         )
# 
#     def forward(self, kspace_us):#(self, kspace_us, mask):
#         x1 = self.conv_block1(kspace_us)
#         x2 = self.conv_block2(x1)
#         x3 = self.conv_block3(x2)
#         x4 = self.conv_block4(x3)
#         x5 = self.up_block1(x4)
#         x6 = self.up_block2(x5)
#         x7 = self.up_block3(x6)
#         #x8 = torch.mul(x7, mask)
#         x9 = self.final_block(x7) #x8
#         return x9



#
# class Block(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
#         self.relu  = nn.ReLU()
#         self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         return self.conv2(self.relu(self.conv1(x)))
#
#
# class Encoder(nn.Module):
#     def __init__(self, chs=(1,32,64,128,256)):
#         super().__init__()
#         self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
#         self.pool       = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv       = nn.Conv2d(chs[-1], chs[-1], kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         ftrs = []
#         for block in self.enc_blocks:
#             x = block(x)
#             ftrs.append(x)
#             x = self.pool(x)
#
#         x = self.conv(x)
#         return x, ftrs
#
#
# class Decoder(nn.Module):
#     def __init__(self, chs=(512, 256, 128, 64,32,32)):
#         super().__init__()
#         self.chs         = chs
#         # self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
#         self.upsample    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+2]) for i in range(len(chs)-2)])
#
#
#     def forward(self, x, encoder_features):
#         for i in range(len(self.chs)-2):
#             x        = self.upsample(x)
#             # enc_ftrs = self.crop(encoder_features[i], x)
#             x        = torch.cat([x, encoder_features[i]], dim=1)
#             x        = self.dec_blocks[i](x)
#         return x
#
#     def crop(self, enc_ftrs, x):
#         _, _, H, W = x.shape
#         enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
#         return enc_ftrs
#
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         return self.conv(x)
#
# class UNet(nn.Module):
#     def __init__(self, enc_chs=(1,32,64,128,256), dec_chs=(512, 256, 128, 64,32,32)):
#         super().__init__()
#         self.encoder     = Encoder(enc_chs)
#         self.decoder     = Decoder(dec_chs)
#         self.output_16   = OutConv(32,16)
#         self.output_1    = OutConv(16,1)
#
#     def forward(self, x):
#         enc, enc_ftrs = self.encoder(x)
#         dec      = self.decoder(enc, enc_ftrs[::-1][:])
#         out_16 = self.output_16(dec)
#         out = self.output_1(out_16)
#
#         return out



# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UNet, self).__init__()
#
#         self.conv_block1 = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#         self.conv_block2 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#         self.up_block = nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#
#         self.final_block = nn.Sequential(
#             nn.Conv2d(64, out_channels, 1),
#             nn.Tanh()
#         )
#
#     def forward(self, kspace_us, mask):
#         x1 = self.conv_block1(kspace_us)
#         print(x1.shape, '#####')
#         x2 = self.conv_block2(x1)
#         print(x2.shape)
#         x3 = self.up_block(x2)
#         print(x3.shape)
#         x4 = torch.mul(x3, mask)
#         x5 = self.final_block(x4)
#         return x5
#
