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
                                  stride = 1, padding = 1, bias = False)
        self.bn_2 = nn.BatchNorm3d(self.out_channel)
        self.w_3 = nn.Conv3d(self.out_channel, self.out_channel, kernel_size = 3, \
                                  stride = 1, padding = 1, bias = False)
        self.bn_3 = nn.BatchNorm3d(self.out_channel)
        self.w_4 = nn.Conv3d(self.out_channel, self.out_channel, kernel_size = 3, \
                                  stride = 1, padding = 1, bias = False)
        self.bn_4 = nn.BatchNorm3d(self.out_channel)
        self.w_5 = nn.Conv3d(self.out_channel, self.out_channel, kernel_size = 3, \
                                  stride = 1, padding = 1, bias = False)
        self.bn_5 = nn.BatchNorm3d(self.out_channel)
        self.relu = nn.ReLU()

        for w in self.modules():
            if isinstance(w, nn.Conv3d):
                getattr(nn.init, self.init_method)(w.weight)

    def forward(self, input): 
        step_1 = self.bn_1(self.w_1(input))
        step_2 = self.relu(self.bn_2(self.w_2(step_1)))
        step_3 = self.bn_3(self.w_3(step_2))
        step_3 = self.relu(step_3 + step_1)
        step_4 = self.relu(self.bn_4(self.w_4(step_3)))
        step_5 = self.bn_5(self.w_5(step_4))
        step_5 = self.relu(step_5 + step_3)
        return step_5

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
        
        self.w_1 = nn.Conv3d(self.input_channels, self.output_channels_1, (3,7,7), \
                                  stride=(1,2,2), padding = (1,3,3), bias = False)  #why false
        self.bn = nn.BatchNorm3d(self.output_channels_1)
        self.relu = nn.ReLU()
        self.res_2 = ResBlock(self.output_channels_1, self.output_channels_2, stride = 1, \
                              init_method = 'kaiming_normal_')
        self.res_3 = ResBlock(self.output_channels_2, self.output_channels_3, stride = 2, \
                              init_method = 'kaiming_normal_')
        self.res_4 = ResBlock(self.output_channels_3, self.output_channels_4, stride = 1, \
                              init_method = 'kaiming_normal_')

        for w in self.modules():
            if isinstance(w, nn.Conv3d):
                getattr(nn.init, self.init_method)(w.weight)

    def forward(self, input): 
        block1 = self.relu(self.bn(self.w_1(input)))
        
        block2 = self.res_2(block1)
        block3 = self.res_3(block2)
        block4 = self.res_4(block3)
        return block4
        
if __name__ == '__main__':

    # gradient check    
    b_s = 8    
    outchannel = 4
    seq_length = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res3D = Res3D(input_channels=16, output_channels=[16, 16, 32, 4], init_method = 'kaiming_normal_')
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(2, 16, 10, 112, 112))
    target = Variable(torch.randn(2, 4, 5, 7, 7)).double()

    # view parameter: don't include batch size
    
    summary(res3D, input_size=(16, 10, 112, 112))   
    output = res3D(input)
    output = output[0][0].double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)
