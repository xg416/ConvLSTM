# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:38:03 2019

@author: Xg Zhang
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary

class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride = 1, init_method = 'kaiming_normal_'):
        super(ResBlock, self).__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        self.init_method = init_method
        self.stride = stride

        self.w_1 = nn.Conv3d(self.in_channel, self.out_channel, kernel_size = 1, \
                                  stride=self.stride, padding = 0, bias = False)
        self.bn_1 = nn.BatchNorm3d(self.out_channel)
        self.w_2 = nn.Conv3d(self.out_channel, self.out_channel, kernel_size = 3, \
                                  stride=self.stride, padding = 1, bias = False)
        self.bn_2 = nn.BatchNorm3d(self.out_channel)
        self.w_3 = nn.Conv3d(self.out_channel, self.out_channel, kernel_size = 3, \
                                  stride=self.stride, padding = 1, bias = False)
        self.bn_3 = nn.BatchNorm3d(self.out_channel)
        self.w_4 = nn.Conv3d(self.out_channel, self.out_channel, kernel_size = 3, \
                                  stride=self.stride, padding = 1, bias = False)
        self.bn_4 = nn.BatchNorm3d(self.out_channel)
        self.w_5 = nn.Conv3d(self.out_channel, self.out_channel, kernel_size = 3, \
                                  stride=self.stride, padding = 1, bias = False)
        self.bn_5 = nn.BatchNorm3d(self.out_channel)
        self.relu = nn.ReLU()

        for w in self.modules():
            if isinstance(w, nn.Conv3d):
                getattr(nn.init, self.init_method)(w.weight)

        def forward(self, input): 
            step_1 = self.bn_1(self.w_1(input))
            step_2 = self.relu(self.bn_2(self.w_2(step_1)))
            



class Res3D(nn.Module):
    def __init__(self, input_channels, output_channels, init_method = 'kaiming_normal_'):
        super(Res3D, self).__init__()
        
        assert len(output_channels) == 4
        self.input_channels = input_channels
        self.output_channels_1 = output_channels[0]
        self.output_channels_2 = output_channels[1]
        self.output_channels_3 = output_channels[2]
        self.output_channels_4 = output_channels[3]
        self.init_method = init_method
        
        self.w_1_1 = nn.Conv3d(self.input_channels, self.output_channels_1, (3,7,7), \
                                  stride=(1,2,2), padding = (1,3,3), bias = False)  #why false
        self.Batch1_1 = nn.BatchNorm3d(self.output_channels_1)
        self.w_2_1 = nn.Conv3d(self.output_channels_1, self.output_channels_2, kernel_size = 1, \
                                  stride=1, padding = 0, bias = False)
        self.w_2_2 = nn.Conv3d(self.output_channels_2, self.output_channels_2, kernel_size = 3, \
                                  stride=1, padding = 1, bias = False)
        self.w_2_3 = nn.Conv3d(self.output_channels_2, self.output_channels_2, kernel_size = 3, \
                                  stride=1, padding = 1, bias = False)
        self.w_2_4 = nn.Conv3d(self.output_channels_2, self.output_channels_2, kernel_size = 3, \
                                  stride=1, padding = 1, bias = False)
        self.w_2_5 = nn.Conv3d(self.output_channels_2, self.output_channels_2, kernel_size = 3, \
                                  stride=1, padding = 1, bias = False)
        self.w_3_1 = nn.Conv3d(self.output_channels_2, self.output_channels_3, kernel_size = 1, \
                                  stride=2, padding = 0, bias = False)
        self.w_3_2 = nn.Conv3d(self.output_channels_3, self.output_channels_3, kernel_size = 3, \
                                  stride=2, padding = 1, bias = False)
        self.w_3_3 = nn.Conv3d(self.output_channels_3, self.output_channels_3, kernel_size = 3, \
                                  stride=1, padding = 1, bias = False)
        self.w_3_4 = nn.Conv3d(self.output_channels_3, self.output_channels_3, kernel_size = 3, \
                                  stride=1, padding = 1, bias = False)
        self.w_3_5 = nn.Conv3d(self.output_channels_3, self.output_channels_3, kernel_size = 3, \
                                  stride=1, padding = 1, bias = False)
        self.w_4_1 = nn.Conv3d(self.output_channels_3, self.output_channels_4, kernel_size = 1, \
                                  stride=1, padding = 0, bias = False)
        self.w_4_2 = nn.Conv3d(self.output_channels_4, self.output_channels_4, kernel_size = 3, \
                                  stride=1, padding = 1, bias = False)
        self.w_4_3 = nn.Conv3d(self.output_channels_4, self.output_channels_4, kernel_size = 3, \
                                  stride=1, padding = 1, bias = False)
        self.w_4_4 = nn.Conv3d(self.output_channels_4, self.output_channels_4, kernel_size = 3, \
                                  stride=1, padding = 1, bias = False)
        self.w_4_5 = nn.Conv3d(self.output_channels_4, self.output_channels_4, kernel_size = 3, \
                                  stride=1, padding = 1, bias = False)

        for w in self.modules():
            if isinstance(w, nn.Conv3d):
                getattr(nn.init, self.init_method)(w.weight)

    def forward(self, input): 
        block1_1 = self.w_1_1(input)
        block1_1 = self.Batch1(block1_1)
        block1_1 = nn.ReLU()(block1_1)
        
        block2_1 = self.w_2_1(block1_1)
        block2_1 = nn.BatchNorm3d(self.output_channels_2)(block2_1)
        block2_2 = self.w_2_2(block2_1)
        block2_2 = nn.BatchNorm3d(self.output_channels_2)(block2_1)
        block2_2 = nn.ReLU()(block2_2)
        block2_3 = self.w_2_3(block2_2)
        block2_3 = nn.BatchNorm3d(self.output_channels_2)(block2_3)
        block2_3 = nn.ReLU()(block2_3 + block2_1)
        block2_4 = self.w_2_4(block2_3)
        block2_4 = nn.BatchNorm3d(self.output_channels_2)(block2_4)
        block2_4 = nn.ReLU()(block2_4)
        block2_5 = self.w_2_5(block2_4)
        block2_5 = nn.BatchNorm3d(self.output_channels_2)(block2_5)
        block2_5 = nn.ReLU()(block2_5 + block2_3)

        block3_1 = self.w_3_1(block2_5)
        block3_1 = nn.BatchNorm3d(self.output_channels_3)(block3_1)
        block3_2 = self.w_3_2(block3_1)
        block3_2 = nn.BatchNorm3d(self.output_channels_3)(block3_2)
        block3_2 = nn.ReLU()(block3_1)
        block3_3 = self.w_3_3(block3_2)
        block3_3 = nn.BatchNorm3d(self.output_channels_3)(block3_3)
        block3_3 = nn.ReLU()(block3_3 + block3_1)
        block3_4 = self.w_3_4(block3_3)
        block3_4 = nn.BatchNorm3d(self.output_channels_3)(block3_4)
        block3_4 = nn.ReLU()(block3_4)
        block3_5 = self.w_3_5(block3_4)
        block3_5 = nn.BatchNorm3d(self.output_channels_3)(block3_5)
        block3_5 = nn.ReLU()(block3_5 + block3_3)

        block4_1 = self.w_4_1(block3_5)
        block4_1 = nn.BatchNorm3d(self.output_channels_4)(block4_1)
        block4_2 = self.w_4_2(block4_1)
        block4_2 = nn.BatchNorm3d(self.output_channels_4)(block4_2)
        block4_2 = nn.ReLU()(block4_2)
        block4_3 = self.w_4_3(block4_2)
        block4_3 = nn.BatchNorm3d(self.output_channels_4)(block4_3)
        block4_3 = nn.ReLU()(block4_3 + block4_1)
        block4_4 = self.w_4_4(block4_3)
        block4_4 = nn.BatchNorm3d(self.output_channels_4)(block4_4)
        block4_4 = nn.ReLU()(block4_4)
        block4_5 = self.w_4_5(block4_4)
        block4_5 = nn.BatchNorm3d(self.output_channels_4)(block4_5)
        block4_5 = nn.ReLU()(block4_5 + block4_3)

        return block4_5
        
if __name__ == '__main__':

    # gradient check    
    b_s = 8    
    outchannel = 4
    seq_length = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res3D = Res3D(input_channels=16, output_channels=[16, 16, 32, 4], init_method = 'kaiming_normal_').to(device)
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(8, 16, 10, 28, 28)).cuda()
    target = Variable(torch.randn(8, 4, 5, 7, 7)).double().cuda()
    k = 0
    for i in res3D.parameters():
        k = k+1
        print(k)
    # view parameter: don't include batch size
    
    #summary(res3D, input_size=(16, 32, 112, 112))   
    output = res3D(input)
    output = output[0][0].double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)
