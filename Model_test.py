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
import time

class Res_clstm_MN(nn.Module):
    def __init__(self, input_shape, number_class, AttenMethod):
        super(Res_clstm_MN, self).__init__()
        self.in_channels = input_shape[1]
        self.OutChannel_Res3D = [64, 64, 128, 256]
        self.in_LSTM = self.OutChannel_Res3D[-1]
        self.hidden_LSTM = [256, 256]
        self.in_channel_MN = self.hidden_LSTM[-1]
        self.input_shape = input_shape
        self.step_lstm = int(input_shape[2] / 2)
        self.batch_size = input_shape[0]
        self.number_class = number_class

        self.r3D = Res3D(self.in_channels, output_channels=self.OutChannel_Res3D, \
                  init_method = 'kaiming_normal_')
        self.aclstm = AttenConvLSTM(input_channels=self.in_LSTM, hidden_channels = self.hidden_LSTM, \
                                 kernel_size=3, step = self.step_lstm, init_method = 'xavier_normal_',\
                                 AttenMethod = AttenMethod)
        self.MoNet = MobileNet(input_channels=self.in_channel_MN)
        self.avgpool3d = nn.AvgPool3d(kernel_size = (self.step_lstm, 4, 4))
        self.dense = nn.Linear(1024, self.number_class)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, input): 
        res = self.r3D(input)
        convlstm = self.aclstm(res)
        mn = self.MoNet(convlstm)
        gpooling = self.avgpool3d(mn)
        flatten = gpooling.view(self.batch_size, -1)
        flatten = self.dense(flatten)
   #     result = self.softmax(flatten)
        
        return flatten

if __name__ == '__main__':
    seq_length = 32
    classes = 250
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()
    input = Variable(torch.randn(2, 3, seq_length, 112, 112)).to(device)
    target = Variable(torch.randn(2, classes)).argmax().to(device)
    
    Model = Res_clstm_MN(input_shape = input.shape, \
        number_class = classes, AttenMethod = 'd').to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    output = Model(input)
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-5, raise_exception=True)
    end = time.time()
    print(res, end-start) 
