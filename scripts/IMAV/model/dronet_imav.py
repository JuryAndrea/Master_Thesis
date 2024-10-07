# Copyright (C) 2020 ETH Zurich, Switzerland
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.apache.md in the top directory for details.
# You may obtain a copy of the License at

    # http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File:    dronet_v2.py
# Original author:  Daniele Palossi <dpalossi@iis.ee.ethz.ch>
# Author:           Lorenzo Lamberti <lorenzo.lamberti@unibo.it>
# Date:    5.1.2021

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import nemo
import random

################################################################################
# PULP-Dronet building blocks #
################################################################################
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU6(inplace=False)
        self.relu2 = nn.ReLU6(inplace=False)
        self.bypass = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn_bypass = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU6(inplace=False)
        self.add = nemo.quant.pact.PACT_IntegerAdd()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x_bypass = self.bypass(identity)
        x_bypass = self.bn_bypass(x_bypass)
        x_bypass = self.relu3(x_bypass)
        x = self.add(x, x_bypass)
        return x


################################################################################
# IMAV #
################################################################################

class dronet_imav(nn.Module):
    def __init__(self, depth_mult=1.0, image_size=[324,324],fc_in_size=[11,11], block_class=ResBlock, stride=[2,2,2,2,2], outputs=7, nemo=False):
        super(dronet_imav, self).__init__()
        self.nemo=nemo # Prepare network for quantization? [True, False]
        first_conv_channels=int(32*depth_mult)
        #conv 5x5, 1, 32, 200x200, /2
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=first_conv_channels, kernel_size=5, stride=stride[0], padding=2, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU6(inplace=False) # chek with 3 different
        #max pooling 2x2, 32, 32, 100x100, /2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=stride[1], padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.resBlock1 = block_class(first_conv_channels, first_conv_channels, stride=stride[2])
        self.resBlock2 = block_class(first_conv_channels, first_conv_channels*2,stride=stride[3])
        self.resBlock3 = block_class(first_conv_channels*2, first_conv_channels*4,stride=stride[4])
        if not self.nemo: self.dropout = nn.Dropout(p=0.5, inplace=False)

        # fc_in_size =  np.round(image_size/np.prod(stride)).astype(int) # [11,11]
        fc_in_size=fc_in_size
        fc_size = fc_in_size[0] * fc_in_size[1] * (first_conv_channels*4)
        self.fc = nn.Linear(in_features=fc_size, out_features=outputs, bias=False)
        if not self.nemo: 
            self.sig = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        if not self.nemo: x = self.dropout(x)
        x = x.flatten(1)
        
        x = self.fc(x)
        # outputs: [edge, no_edge, corner, yaw_target, left_pcoll, center_pcoll. right_pcoll]
        edge = x[:, [0,1,2]]
        target_yaw = x[:,3]
        pcoll = x[:, [4,5,6]]
        if not self.nemo:
            edge = self.softmax(edge)
            pcoll[:,0] = self.sig(pcoll[:,0])
            pcoll[:,1] = self.sig(pcoll[:,1])
            pcoll[:,2] = self.sig(pcoll[:,2])
        return [edge, target_yaw, pcoll]


################################################################################
# IMAV with MID FUSION approach of the ultrasonic scalar value #
################################################################################
class dronet_imav_v2(nn.Module):
    def __init__(self, depth_mult=1.0, image_size=[324,324],fc_in_size=[11,11], block_class=ResBlock, stride=[2,2,2,2,2], outputs=7, nemo=False):
        super(dronet_imav_v2, self).__init__()
        self.nemo=nemo # Prepare network for quantization? [True, False]
        first_conv_channels=int(32*depth_mult)
        
        out_fc4_size = 4
        
        #conv 5x5, 1, 32, 200x200, /2
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=first_conv_channels, kernel_size=5, stride=stride[0], padding=2, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU6(inplace=False) # chek with 3 different
        #max pooling 2x2, 32, 32, 100x100, /2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=stride[1], padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.resBlock1 = block_class(first_conv_channels, first_conv_channels, stride=stride[2])
        self.resBlock2 = block_class(first_conv_channels, first_conv_channels*2,stride=stride[3])
        
        self.resBlock3 =block_class(first_conv_channels*2 + out_fc4_size, first_conv_channels*4,stride=stride[4])
        if not self.nemo: self.dropout = nn.Dropout(p=0.5, inplace=False)

        # fc_in_size =  np.round(image_size/np.prod(stride)).astype(int) # [11,11]
        fc_in_size=fc_in_size
        fc_size = fc_in_size[0] * fc_in_size[1] * (first_conv_channels*4)
        self.fc = nn.Linear(in_features=fc_size, out_features=outputs, bias=False)
        if not self.nemo: 
            self.sig = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=1)
        
        # Sensor Fusion MLP
        self.fc2 = nn.Linear(in_features=1, out_features=64, bias=False)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=64, bias=False)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=64, out_features=out_fc4_size, bias=False)
        self.relu4 = nn.ReLU()
        

    def forward(self, x, y):
        
        # x is the image, y is the ultrasonic scalar value
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        
        # Mid Fusion MLP
        y = y.view(y.size(0), 1)
        y = self.relu2(self.fc2(y))
        y = self.relu3(self.fc3(y))
        y = self.relu4(self.fc4(y))
        y = y.view(y.size(0), 4, 1, 1)
        y = y.expand(y.size(0), 4, 10, 10)
        x = torch.cat((x, y), dim=1)
        
        x = self.resBlock3(x)
        
        if not self.nemo: x = self.dropout(x)
        x = x.flatten(1)

        x = self.fc(x)    
        
        # outputs: [edge, no_edge, corner, yaw_target, left_pcoll, center_pcoll. right_pcoll]
        edge = x[:, [0,1,2]]
        target_yaw = x[:,3]
        pcoll = x[:, [4,5,6]]
        if not self.nemo:
            edge = self.softmax(edge)
            pcoll[:,0] = self.sig(pcoll[:,0])
            pcoll[:,1] = self.sig(pcoll[:,1])
            pcoll[:,2] = self.sig(pcoll[:,2])
        return [edge, target_yaw, pcoll]


################################################################################
# IMAV with LATE FUSION approach of the ultrasonic scalar value #
################################################################################
class dronet_imav_v3(nn.Module):
    def __init__(self, depth_mult=1.0, image_size=[324,324],fc_in_size=[11,11], block_class=ResBlock, stride=[2,2,2,2,2], outputs=7, nemo=False):
        super(dronet_imav_v3, self).__init__()
        self.nemo=nemo # Prepare network for quantization? [True, False]
        first_conv_channels=int(32*depth_mult)
        #conv 5x5, 1, 32, 200x200, /2
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=first_conv_channels, kernel_size=5, stride=stride[0], padding=2, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU6(inplace=False) # chek with 3 different
        #max pooling 2x2, 32, 32, 100x100, /2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=stride[1], padding=0, dilation=1, return_indices=False, ceil_mode=False)
        
        self.resBlock1 = block_class(first_conv_channels, first_conv_channels, stride=stride[2])
        self.resBlock2 = block_class(first_conv_channels, first_conv_channels*2,stride=stride[3])
        self.resBlock3 = block_class(first_conv_channels*2, first_conv_channels*4,stride=stride[4])
        
        if not self.nemo: self.dropout = nn.Dropout(p=0.5, inplace=False)

        # fc_in_size =  np.round(image_size/np.prod(stride)).astype(int) # [11,11]
        fc_in_size=fc_in_size
        out_fc4_size = 4
        fc_size = fc_in_size[0] * fc_in_size[1] * (first_conv_channels*4) + out_fc4_size
        self.fc = nn.Linear(in_features=fc_size, out_features=outputs, bias=False)
        if not self.nemo: 
            self.sig = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=1)
        
        # Sensor Fusion MLP
        self.fc2 = nn.Linear(in_features=1, out_features=64, bias=False)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=64, bias=False)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=64, out_features=out_fc4_size, bias=False)
        self.relu4 = nn.ReLU()
        

    def forward(self, x, y):
        
        # x is the image, y is the ultrasonic scalar value
        
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)

        if not self.nemo: x = self.dropout(x)
        x = x.flatten(1)
        
        # Late Fusion MLP
        y = y.view(y.size(0), 1)
        y = self.relu2(self.fc2(y))
        y = self.relu3(self.fc3(y))
        y = self.relu4(self.fc4(y))
        x = torch.cat((x, y), dim=1)
        
        x = self.fc(x)    
        
        # outputs: [edge, no_edge, corner, yaw_target, left_pcoll, center_pcoll. right_pcoll]
        edge = x[:, [0,1,2]]
        target_yaw = x[:,3]
        pcoll = x[:, [4,5,6]]
        if not self.nemo:
            edge = self.softmax(edge)
            pcoll[:,0] = self.sig(pcoll[:,0])
            pcoll[:,1] = self.sig(pcoll[:,1])
            pcoll[:,2] = self.sig(pcoll[:,2])
        return [edge, target_yaw, pcoll]