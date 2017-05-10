# Taken from https://github.com/ikostrikov/pytorch-a3c
from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    """
    Not actually using this but let's keep it here in case that changes
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ES(torch.nn.Module):

    def __init__(self, num_inputs, action_space,
                 small_net=False, use_lstm=False, use_a3c_net=False, 
                 use_virtual_batch_norm=False):
        """
        Really I should be using inheritance for the small_net here
        """
        super(ES, self).__init__()
        num_outputs = action_space.n
        self.small_net = small_net
        self.use_lstm = use_lstm
        self.use_virtual_batch_norm = use_virtual_batch_norm
        self.a3c_net = use_a3c_net
        if self.small_net:
            self.linear1 = nn.Linear(num_inputs, 64)
            self.linear2 = nn.Linear(64, 64)
            self.actor_linear = nn.Linear(64, num_outputs)
        elif self.a3c_net:
            self.conv1 = nn.Conv2d(num_inputs, 16, 8, stride=4)
            self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
            if self.use_virtual_batch_norm:
                self.bn1 = nn.BatchNorm2d(16, affine=False, momentum=1.)
                self.bn2 = nn.BatchNorm2d(32, affine=False, momentum=1.)
            else:
                self.bn1 = lambda x: x
                self.bn2 = lambda x: x
            self.fc = nn.Linear(32*3*3, 256)
            self.actor_linear = nn.Linear(256, num_outputs)
        else:
            self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            if self.use_virtual_batch_norm:
                self.bn1 = nn.BatchNorm2d(32, affine=False, momentum=1.)
                self.bn2 = nn.BatchNorm2d(32, affine=False, momentum=1.)
                self.bn3 = nn.BatchNorm2d(32, affine=False, momentum=1.)
                self.bn4 = nn.BatchNorm2d(32, affine=False, momentum=1.)
            else:
                self.bn1 = lambda x: x
                self.bn2 = lambda x: x
                self.bn3 = lambda x: x
                self.bn4 = lambda x: x
            if self.use_lstm:
                self.lstm = nn.LSTMCell(32*3*3, 256)
            else:
                self.fc = nn.Linear(32*3*3, 256)
            self.actor_linear = nn.Linear(256, num_outputs)
        self.train(False)
    
    def forward(self, inputs):
        if self.small_net:
            x = F.elu(self.linear1(inputs))
            x = F.elu(self.linear2(x))
            return self.actor_linear(x)
        elif self.a3c_net:
            x = F.relu(self.bn1(self.conv1(inputs)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = x.view(-1, 32*3*3)
            x = self.fc(x)
            return F.softmax(self.actor_linear(x))
        else:
            if self.use_lstm:
                inputs, (hx, cx) = inputs
            x = F.elu(self.bn1(self.conv1(inputs)))
            x = F.elu(self.bn2(self.conv2(x)))
            x = F.elu(self.bn3(self.conv3(x)))
            x = F.elu(self.bn4(self.conv4(x)))
            x = x.view(-1, 32*3*3)
            if self.use_lstm:
                hx, cx = self.lstm(x, (hx, cx))
                x = hx
                return F.softmax(self.actor_linear(x)), (hx, cx)
            else:
                x = self.fc(x)
                return F.softmax(self.actor_linear(x))
                
    def init_virtual_batch_norm(self, batch):
        if not self.use_virtual_batch_norm:
            raise Exception(
                'Network was not constructed for virtual batch normalization.'
                'Construct with use_virtual_batch_norm==True.')
        self.train(True)
        self.forward(batch)
        self.train(False)

    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.data.numpy().flatten().shape[0]
        return count

    def es_params(self):
        """
        The params that should be trained by ES (all of them)
        """
        return [(k, v) for k, v in zip(self.state_dict().keys(),
                                       self.state_dict().values())]
