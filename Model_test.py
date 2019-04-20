# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 20:57:23 2019

@author: Xingguang Zhang
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from Atten_CLSTM import AttenConvLSTM
from Res3D import Res3D
from MobileNet import MobileNet

class MyModule(nn.Module):
    def __init__(self, input_channels, input_shape, number_class):
        super(MyModule, self).__init__()
        self.in_channels = input_channels
        self.OutChannel_Res3D = [64, 64, 128, 256]
        self.in_LSTM = self.OutChannel_Res3D[-1]
        self.hidden_LSTM = [256, 256]
        self.in_channel_MN = self.hidden_LSTM[-1]
        self.input_shape = input_shape
        self.step_lstm = int(input_shape[2] / 2)
        self.batch_size = input_shape[0]
        self.number_class = number_class
    
    def forward(self, input): 
        res = Res3D(self.in_channels, output_channels=self.OutChannel_Res3D, \
                  init_method = 'kaiming_normal_')(input)
        convlstm = AttenConvLSTM(input_channels=self.in_LSTM, hidden_channels = self.hidden_LSTM, \
                                 kernel_size=3, step = self.step_lstm, init_method = 'xavier_normal_',\
                                 AttenMethod = 'b')(res)
        mn = MobileNet(input_channels=self.in_channel_MN)(convlstm)
        gpooling = nn.AvgPool3d(kernel_size = (self.step_lstm, 4, 4))(mn)
        flatten = gpooling.View(self.batch_size, -1)
        flatten = nn.Linear(1024, self.number_class)(flatten)
        result = nn.Softmax(dim = 1)(flatten)
        
        return result

if __name__ == '__main__':
    seq_length = 32
    classes = 250
    
    input = Variable(torch.randn(2, 3, seq_length, 112, 112))
    target = Variable(torch.randn(2, classes)).double()
    
    Model = MyModule(input_channels=3, input_shape = input.shape, number_class = classes)
    loss_fn = torch.nn.MSELoss()
    
    output = Model(input)
    output = output[0][0].double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res) 
