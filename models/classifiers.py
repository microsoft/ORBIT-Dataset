"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file adaptation_networks.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/adaptation_networks.py)
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

from models.mlps import DenseResidualBlock

class LinearClassifierAdapter(nn.Module):
    def __init__(self, d_theta):
        super(LinearClassifierAdapter, self).__init__()
        self.weight_processor = self._make_layer(d_theta, d_theta)
        self.bias_processor = self._make_layer(d_theta, 1)

    @staticmethod
    def _make_layer(in_size, out_size):
        return DenseResidualBlock(in_size, out_size)

    def forward(self, representation_dict):
        classifier_param_dict = {}
        class_weight = []
        class_bias = []

        label_set = list(representation_dict.keys())
        label_set.sort()
        num_classes = len(label_set)

        for class_num in label_set:
            nu = representation_dict[class_num]
            class_weight.append(self.weight_processor(nu))
            class_bias.append(self.bias_processor(nu))

        classifier_param_dict['weight'] = torch.cat(class_weight, dim=0)
        classifier_param_dict['bias'] = torch.reshape(torch.cat(class_bias, dim=1), [num_classes, ])
        return classifier_param_dict

class PrototypicalClassifierAdapter(nn.Module):
    def __init__(self):
        super(PrototypicalClassifierAdapter, self).__init__()

    def forward(self, representation_dict):
        classifier_param_dict = {}
        class_weight = []
        class_bias = []

        label_set = list(representation_dict.keys())
        label_set.sort()
        num_classes = len(label_set)

        for class_num in label_set:
            # equation 8 from the prototypical networks paper
            nu = representation_dict[class_num]
            class_weight.append(2 * nu)
            class_bias.append((-torch.matmul(nu, nu.t()))[None, None])

        classifier_param_dict['weight'] = torch.cat(class_weight, dim=0)
        classifier_param_dict['bias'] = torch.reshape(torch.cat(class_bias, dim=1), [num_classes, ])

        return classifier_param_dict

def linear_classifier(x, param_dict):
    return F.linear(x, param_dict['weight'], param_dict['bias'])
