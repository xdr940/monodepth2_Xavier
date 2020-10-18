# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import numpy as np
import torch.nn as nn
import torch

from collections import OrderedDict
from networks.layers import ConvBlock,Conv3x3,upsample


class DepthDecoder2(nn.Module):
    def __init__(self, num_ch_enc=[64, 64, 128, 256, 512], scales=range(0), num_output_channels=1, use_skips=True):
        super(DepthDecoder2, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
                                                                #0   1   2   3    4
        self.num_ch_enc = num_ch_enc                          #[64, 64, 128, 256, 512] #skip features
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])#[16, 32, 64, 128, 256] #decoder layer out channels
                                                            #0   1   2   3    4
        # decoder
        self.convs = OrderedDict()

        #self.convs[("upconv", 4, 0)]

        #[64,64,128,256,512]


        # self.convs[("upconv", 4, 0)] = ConvBlock(self.num_ch_enc[4],                    self.num_ch_enc[3])
        # self.convs[("upconv", 4, 1)] = ConvBlock(2*self.num_ch_enc[3],                  self.num_ch_dec[4])
        #
        # self.convs[("upconv", 3, 0)] = ConvBlock(self.num_ch_dec[4],                    self.num_ch_enc[2])
        # self.convs[("upconv", 3, 1)] = ConvBlock(2*self.num_ch_enc[2],                  self.num_ch_dec[3])
        #
        # self.convs[("upconv", 2, 0)] = ConvBlock(self.num_ch_dec[3],                    self.num_ch_enc[1])
        # self.convs[("upconv", 2, 1)] = ConvBlock(2*self.num_ch_enc[1],                  self.num_ch_dec[2])
        #
        # self.convs[("upconv", 1, 0)] = ConvBlock(self.num_ch_dec[2],                    self.num_ch_enc[0])
        # self.convs[("upconv", 1, 1)] = ConvBlock(2*self.num_ch_enc[0],                  self.num_ch_dec[1])
        #
        # self.convs[("upconv", 0, 0)] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])
        # self.convs[("upconv", 0, 1)] = ConvBlock(self.num_ch_enc[0], self.num_ch_dec[0])

        # just copy

        self.convs[("upconv", 4, 0)] = ConvBlock(512,256)
        self.convs[("upconv", 4, 1)] = ConvBlock(512, 256)

        self.convs[("upconv", 3, 0)] = ConvBlock(256, 128)
        self.convs[("upconv", 3, 1)] = ConvBlock(256, 128)

        self.convs[("upconv", 2, 0)] = ConvBlock(128, 64)
        self.convs[("upconv", 2, 1)] = ConvBlock(128, 64)

        self.convs[("upconv", 1, 0)] = ConvBlock(64, 32)
        self.convs[("upconv", 1, 1)] = ConvBlock(96, 32)

        self.convs[("upconv", 0, 0)] = ConvBlock(32, 16)
        self.convs[("upconv", 0, 1)] = ConvBlock(16, 16)




        self.convs[("dispconv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, f0,f1,f2,f3,f4):
        self.outputs = {}

        # decoder
        x = f4

        #i = 4
        x = self.convs[("upconv", 4, 0)](x)
        x = [upsample(x)]
        if self.use_skips:
            x += [f3]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 4, 1)](x)

        #i=3
        x = self.convs[("upconv", 3, 0)](x)
        x = [upsample(x)]
        if self.use_skips:
            x += [f2]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 3, 1)](x)

        # i =2
        x = self.convs[("upconv", 2, 0)](x)
        x = [upsample(x)]
        if self.use_skips:
            x += [f1]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 2, 1)](x)

        # i =1
        x = self.convs[("upconv", 1, 0)](x)
        x = [upsample(x)]
        if self.use_skips:
            x += [f0]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 1, 1)](x)

        # i = 0
        x = self.convs[("upconv", 0, 0)](x)
        x = upsample(x)
        x = self.convs[("upconv", 0, 1)](x)

        ret = self.sigmoid(self.convs[("dispconv", 0)](x))



        return ret

