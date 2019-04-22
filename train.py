# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 20:57:23 2019

@author: Xingguang Zhang
"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from Res_clstm_MN import Res_clstm_MN
import time
from datagen import isoImageGenerator
import inputs as data

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=10, power=1.5):
    """
    Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def success(output, target):
    success_number = 0
    predict = output.argmax(dim = 1)
    predict = np.asarray(predict.cpu())
    target = np.asarray(target.cpu())
    for i in range(predict.size):
        if predict[i] == target[i]:
            success_number += 1
    return success_number

def train(training_datalist, batch_size, seq_len, num_classes, cfg_modality, device, max_iter = 10, current = 0):
    AvgLoss = 0
    batch_num = len(training_datalist[0]) / batch_size
    success_number = 0
    for data, label in isoImageGenerator(training_datalist, batch_size, seq_len, num_classes, cfg_modality, Training = True):
        lr = poly_lr_scheduler(optimizer, init_lr = 0.0001, iter = current, lr_decay_iter=1,max_iter=10, power=1.5)
        data = data.transpose(0,4,1,2,3)
        input = torch.from_numpy(data).float().to(device)
        target = torch.from_numpy(label).argmax(dim = 1).to(device)
        optimizer.zero_grad()
        output = Model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        s =success(output, target)
        success_number += s
        AvgLoss += float(loss)
        print(s, float(loss))
    return AvgLoss / batch_num, success_number / batch_num

def validate(valid_datalist, batch_size, seq_len, num_classes, cfg_modality, device):
    AvgLoss = 0
    batch_num = len(valid_datalist[0]) / batch_size
    success_number = 0
    for data, label in isoImageGenerator(valid_datalist, batch_size, seq_len, num_classes, cfg_modality, Training = False):
        data = data.transpose(0,4,1,2,3)
        input = torch.from_numpy(data).float().to(device)
        target = torch.from_numpy(label).argmax(dim = 1).to(device)
        optimizer.zero_grad()
        output = Model(input)
        loss = loss_fn(output, target)
        success_number += success(output, target)
        AvgLoss += float(loss)
    return AvgLoss / batch_num, success_number / batch_num


if __name__ == '__main__':

    rgb_train_datalist = '/home/isat-deep/Documents/isoGD/train_rgb_list.txt'
    depth_train_datalist = '/home/isat-deep/Documents/isoGD/train_depth_list.txt'
    flow_train_datalist = '/home/isat-deep/Documents/isoGD/train_flow_list.txt'
    rgb_valid_datalist = '/home/isat-deep/Documents/isoGD/valid_rgb_list.txt'
    depth_valid_datalist = '/home/isat-deep/Documents/isoGD/valid_depth_list.txt'
    flow_valid_datalist = '/home/isat-deep/Documents/isoGD/valid_flow_list.txt'
    training_datalist = []
    valid_datalist = []
    model_path = '/home/isat-deep/Documents/isoGD/model.pkl'
    # Modality
    RGB = 0
    Depth = 1
    Flow = 2
    input_channels = 0
    cfg_modality = [RGB, Depth]
    if 0 in cfg_modality:
        input_channels += 3
        training_datalist.append(rgb_train_datalist)
        valid_datalist.append(rgb_valid_datalist)
    if 1 in cfg_modality:
        input_channels += 3
        training_datalist.append(depth_train_datalist)
        valid_datalist.append(depth_valid_datalist)
    if 2 in cfg_modality:
        input_channels += 3
        training_datalist.append(flow_train_datalist)
        valid_datalist.append(flow_valid_datalist)

    batch_size = 8
    seq_len = 32
    num_classes = 249
    total_epoch = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model = Res_clstm_MN(input_shape = (batch_size, input_channels, seq_len, 112, 112), \
        number_class = num_classes, AttenMethod = 'a').to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(Model.parameters(), lr = 0.0001,\
                                momentum = 0.9,\
                                weight_decay = 0.00005)
    cond_train = []
    cond_validate = []
    for epoch in range(total_epoch):
        loss_train, accu_train = train(training_datalist, batch_size, seq_len, num_classes, \
            cfg_modality, device, max_iter = total_epoch, current = epoch)
        loss_validation, accu_validation = validate(valid_datalist, batch_size, seq_len, \
            num_classes, cfg_modality, device)

        cond_train.append([loss_train, accu_train])
        cond_validate.append([loss_validation, accu_validation])
        torch.save(Model, model_path)
