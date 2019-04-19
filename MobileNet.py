# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:18:25 2019

@author: Xingguang Zhang
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary

class Separable_conv(nn.Module):
    '''input: (inchannel, outchannel, stride)'''
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Separable_conv, self).__init__()
        
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.conv1 = nn.Conv2d(self.input_channels, self.input_channels, kernel_size = 3, \
                               stride = stride, padding = 1, bias=False, groups = self.input_channels)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(self.input_channels)
        self.conv2 = nn.Conv2d(self.input_channels, self.output_channels, kernel_size = 1, stride = 1,\
                               padding = 0, bias = False)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn2 = nn.BatchNorm2d(self.output_channels)

    def forward(self, x):
        ''' shape of input data x: (batch, channel, weight, height)'''
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = nn.ReLU()(self.bn2(self.conv2(out)))
        return out

class MobileNet(nn.Module):
    # (256,2) means conv planes=256, conv stride=2, by default conv stride=1
    def __init__(self, input_channels = 256, cfg = [(256,1), (256,2), (512,1), (512,1), \
                              (512,1), (512,1), (512,1), (1024,2), (1024,1)]):
        super(MobileNet, self).__init__()
        self.cfg = cfg
        self.first_channels = input_channels
        self.MoNet = self._make_layers(self.first_channels)

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            outchannels, stride = x
            layers.append(Separable_conv(in_channels, outchannels, stride))
            in_channels = outchannels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x must be in the shape of (batch, channel, weight, height)
        # for 5 input, x must be in the shape of (batch, channel, depth, weight, height)
        # thus depth will be viewed as batch
        inputdim = len(x.shape)
        if inputdim == 5:
            b, c, d, w, h = x.shape
            x = x.permute(0, 2, 1, 3, 4)
            x = x.contiguous().view(-1, c, w, h)
            
        out = self.MoNet(x)
        
        if inputdim == 5:
            _, new_c, new_w, new_h = out.shape
            return out.view(b, d, new_c, new_w, new_h).permute(0, 2, 1, 3, 4)
        else:
            return out

if __name__ == '__main__':

    # gradient check    
    b_s = 8    
    seq_length = 3
    
    mn = MobileNet(input_channels=256)
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(2, 256, seq_length, 28, 28))
    target = Variable(torch.randn(2, 1024, seq_length, 7, 7)).double()
    
    # view parameter: don't include batch size  
    summary(mn, input_size=(256, 10, 28, 28))   
    output = mn(input)
    output = output[0][0].double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)
