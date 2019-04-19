# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:52:29 2019

@author: 10659
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class GateConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(GateConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc_d = nn.Conv2d(self.input_channels, self.input_channels, self.kernel_size, \
                               1, self.padding, bias=True, groups = self.input_channels)
        self.Wxc_p = nn.Conv2d(self.input_channels, self.hidden_channels, (1,1), 1, 0, bias=True)
        self.Whc_d = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size,\
                               1, self.padding, bias=False, groups = self.hidden_channels)
        self.Whc_p = nn.Conv2d(self.hidden_channels, self.hidden_channels, (1,1), 1, 0, bias=True)        
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding,  bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

    def forward(self, x, h, c):
        x_global = nn.AdaptiveAvgPool2d(1)(x)
        h_global = nn.AdaptiveAvgPool2d(1)(h)
        ci = torch.sigmoid(self.Wxi(x_global) + self.Whi(h_global))
        cf = torch.sigmoid(self.Wxf(x_global) + self.Whf(h_global))
        co = torch.sigmoid(self.Wxo(x_global) + self.Who(h_global))
        G = torch.tanh(self.Wxc_p(self.Wxc_d(x)) + self.Whc_p(self.Whc_d(h)))
        cc = cf * c + ci * G
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))


class GateConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, bias=True):
        super(GateConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.bias = bias
        #self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = GateConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = torch.randn(8, self.step, self.hidden_channels[-1], 28, 28)
        for step in range(self.step):
            x = input[:,step,:,:,:]
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i], shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            outputs[:,step,:,:,:] = x
        return outputs, (x, new_c)


if __name__ == '__main__':

    # gradient check    
    outchannel = 4
    seq_length = 16
    convlstm = GateConvLSTM(input_channels=16, hidden_channels=[16, outchannel], kernel_size=3, step = seq_length)
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(8, seq_length, 16, 28, 28))
    target = Variable(torch.randn(8, seq_length, outchannel, 28, 28)).double()
    
    # view parameter: don't include batch size
    
    output = convlstm(input)
    output = output[0][0].double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)
