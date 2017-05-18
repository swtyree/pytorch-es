# Taken from https://github.com/ikostrikov/pytorch-a3c
from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ES(torch.nn.Module):

    def __init__(self, num_inputs, action_space,
                 use_a3c_net=False, use_virtual_batch_norm=False):
        super(ES, self).__init__()
        num_outputs = action_space.n
        self.use_virtual_batch_norm = use_virtual_batch_norm
        self.a3c_net = use_a3c_net
        
        # conv layers
        if self.a3c_net:
            self.conv1 = nn.Conv2d(num_inputs, 16, 8, stride=4)
            self.bnconv1 = nn.BatchNorm2d(16, affine=False, momentum=1.)
            self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
            self.bnconv2 = nn.BatchNorm2d(32, affine=False, momentum=1.)
        else:
            self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
            self.bnconv1 = nn.BatchNorm2d(32, affine=False, momentum=1.)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.bnconv2 = nn.BatchNorm2d(32, affine=False, momentum=1.)
            self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.bnconv3 = nn.BatchNorm2d(32, affine=False, momentum=1.)
            self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.bnconv4 = nn.BatchNorm2d(32, affine=False, momentum=1.)
        
        # fc layers
        self.fc1 = nn.Linear(32*3*3, 256)
        self.bnfc1 = nn.BatchNorm1d(256, affine=False, momentum=1.)
        self.fc2 = nn.Linear(256, num_outputs)
        self.bnfc2 = nn.BatchNorm1d(num_outputs, affine=False, momentum=1.)
        
        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.normal(m.weight.data, std=0.05)
                m.bias.data.zero_()
        
        # set to eval mode
        self.train(False)
    
    def forward(self, inputs):
        if self.a3c_net:
            x = F.relu(self.bnconv1(self.conv1(inputs)))
            x = F.relu(self.bnconv2(self.conv2(x)))
            x = x.view(-1, 32*3*3)
            x = F.relu(self.bnfc1(self.fc1(x)))
            return F.softmax(self.bnfc2(self.fc2(x)))
        else:
            x = F.elu(self.bnconv1(self.conv1(inputs)))
            x = F.elu(self.bnconv2(self.conv2(x)))
            x = F.elu(self.bnconv3(self.conv3(x)))
            x = F.elu(self.bnconv4(self.conv4(x)))
            x = x.view(-1, 32*3*3)
            x = F.elu(self.bnfc1(self.fc1(x)))
            return F.softmax(self.bnfc2(self.fc2(x)))
                
    def do_virtual_batch_norm(self, batch):
        if not self.use_virtual_batch_norm:
            raise Exception(
                'Network was not constructed for virtual batch normalization.'
                'Construct with use_virtual_batch_norm==True.')
        self.train(True)
        self.forward(batch)
        self.train(False)

    def count_parameters(self):
        count = 0
        for _,p in self.get_es_params():
            count += p.data.numel()
        return count

    def get_es_params(self):
        """
        The params that should be trained by ES (all of them)
        """
        return [(k,p) for k,p in self.named_parameters()]
    
    def adjust_es_params(self, multiply=1., add=0.):
        i = 0
        for _,p in self.get_es_params():
            n = p.data.numel()
            p.data *= multiply
            p.data += torch.from_numpy(add[i:i+n]).type(p.data.type())
            i += n
    
    def get_param_norm(self):
        return np.sqrt(sum([p.pow(2).sum().data.numpy() for _,p in self.get_es_params()])[0])
