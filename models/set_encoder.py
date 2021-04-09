"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file set_encoder.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/set_encoder.py)
from the cambridge-mlg/cnaps library (https://github.com/cambridge-mlg/cnaps).

The original license is included below:

Copyright (c) 2019 John Bronskill, Jonathan Gordon, and James Requeima.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.normalisation_layers import TaskNormI 

def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)

class SetEncoder(nn.Module):
    def __init__(self, batch_normalisation):
        super(SetEncoder, self).__init__()
        self.pre_pooling_fn = SimplePrePoolNet(batch_normalisation)
        self.pooling_fn = mean_pooling
        self.post_pooling_fn = Identity()

    def forward(self, x):
        x = self.pre_pooling_fn(x)
        x = self.pooling_fn(x)
        x = self.post_pooling_fn(x)
        return x
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class SimplePrePoolNet(nn.Module):
    def __init__(self, batch_normalisation):
        super(SimplePrePoolNet, self).__init__()
        if batch_normalisation == "task_norm-i":
            self.layer1 = self._make_conv2d_layer_task_norm(3, 64)
            self.layer2 = self._make_conv2d_layer_task_norm(64, 64)
            self.layer3 = self._make_conv2d_layer_task_norm(64, 64)
            self.layer4 = self._make_conv2d_layer_task_norm(64, 64)
            self.layer5 = self._make_conv2d_layer_task_norm(64, 64)
        else:
            self.layer1 = self._make_conv2d_layer(3, 64)
            self.layer2 = self._make_conv2d_layer(64, 64)
            self.layer3 = self._make_conv2d_layer(64, 64)
            self.layer4 = self._make_conv2d_layer(64, 64)
            self.layer5 = self._make_conv2d_layer(64, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    @staticmethod
    def _make_conv2d_layer_task_norm(in_maps, out_maps):
        return nn.Sequential(
            nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1),
            TaskNormI(out_maps),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_maps),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

    def _flatten(self, x):
        sz = x.size()
        return x.view(-1, sz[-3], sz[-2], sz[-1])

    def forward(self, x):
        x = self._flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    @property
    def output_size(self):
        return 64
