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

from models.normalisation_layers import TaskNorm

class SetEncoder(nn.Module):
    """
    Simple set encoder implementing DeepSets (https://arxiv.org/abs/1703.06114). Used for modeling permutation-invariant representations on sets (mainly for extracting task-level embedding of context sets).
    """
    def __init__(self, batch_normalisation):
        """
        Creates an instance of SetEncoder.
        :param batch_normalisation: (str) If "basic", use standard BatchNorm layers, otherwise if "task_norm" use TaskNorm layers.
        :return: Nothing.
        """
        super(SetEncoder, self).__init__()
        self.encoding_fn = SimplePrePoolNet(batch_normalisation)
 
    def forward(self, x):
        """
        Function that encodes each element in x into a 64-dimensonal embedding.
        :param x: (torch.Tensor) Set of elements (for clips it has the shape: batch x clip length x C x H x W).
        :return: (torch.Tensor) Each element in x encoded as a 64-dimensional vector.
        """
        return self.encoding_fn(x)

    def aggregate(self, x, reduction='mean', switch_device=False):
        """
        Function that aggregates the encoded elements in x.
        :param x: (torch.Tensor) Set of encoded elements (i.e. from forward()).
        :param reduction: (str) If 'mean', average the encoded elements in x, otherwise do not average.
        :param switch_device: (bool) If True, move output to second GPU.
        :return: (torch.Tensor) Mean representation of the set as a single vector if reduction = 'mean', otherwise as a set of encoded elements.
        """
        x = torch.cat(x, dim=0)
        if reduction == 'mean':
            x = self.mean_pool(x)
        return x.cuda(1) if switch_device else x

    def mean_pool(self, x):
        """
        Function that mean pools the elements in x.
        :param x: (torch.Tensor) Set of encoded elements (i.e. from forward()).
        :return: (torch.Tensor) Mean representation of the set as a single vector.
        """
        return torch.mean(x, dim=0, keepdim=True)

    @property
    def output_size(self):
        return 64

class SimplePrePoolNet(nn.Module):
    """
    Simple network to encode elements of a set into low-dimensional embeddings. Used before pooling them to obtain a task-level embedding. A multi-layer convolutional network is used, similar to that in https://github.com/cambridge-mlg/cnaps.
    """
    def __init__(self, batch_normalisation):
        """
        Creates an instance of SimplePrePoolNet.
        :param batch_normalisation: (str) If "basic", use standard BatchNorm layers, otherwise if "task_norm" use TaskNorm layers.
        :return: Nothing.
        """
        super(SimplePrePoolNet, self).__init__()
        if batch_normalisation == "task_norm":
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
            TaskNorm(out_maps),
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
        """
        Function that flattens tensor into 4 dimensions if x.dim() >= 5.
        :param x: (torch.Tensor) Set of elements (for clips it has the shape: batch x clip length x C x H x W).
        :return: (torch.Tensor) 4-dimensional version of x.
        """
        sz = x.size()
        return x.view(-1, sz[-3], sz[-2], sz[-1]) if x.dim() >=5 else x

    def forward(self, x):
        """
        Function that encodes each element in x into a 64-dimensonal embedding.
        :param x: (torch.Tensor) Set of elements (for clips it has the shape: batch x clip length x C x H x W).
        :return: (torch.Tensor) Each element in x encoded as a 64-dimensional vector.
        """
        x = self._flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class NullSetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return None

    def aggregate(self, x, reduction='mean', switch_device=False):
        return None
    
    def mean_pool(self, x):
        return None

    @property
    def output_size(self):
        return None
