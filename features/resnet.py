"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file resnet.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/resnet.py)
from the cambridge-mlg/cnaps library (https://github.com/cambridge-mlg/cnaps/)

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
from models.normalisation_layers import TaskNorm, get_normalisation_layer
from feature_adapters.resnet_adaptation_layers import FilmLayer, FilmLayerGenerator

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_fn, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = bn_fn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockFilm(nn.Module):
    """
    Extension to standard ResNet block (https://arxiv.org/abs/1512.03385) with FiLM layer adaptation. After every batch
    normalisation layer, we add a FiLM layer (which applies an affine transformation to each channel in the hidden
    representation). As we are adapting the feature extractor with an external adaptation network, we expect parameters
    to be passed as an argument of the forward pass.
    """
    expansion = 1

    def __init__(self, inplanes, planes, bn_fn, stride=1, downsample=None):
        super(BasicBlockFilm, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = bn_fn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, gamma, beta):
        """
        Implements a forward pass through the FiLM adapted ResNet block. FiLM parameters for adaptation are passed
        through to the method, one gamma / beta set for each convolutional layer in the block (2 for the blocks we are
        working with).
        :param x: (torch.tensor) Batch of images to apply computation to.
        :param gamma: (list::torch.tensor) List of multiplicative FiLM parameter for conv layers (one for each channel).
        :param beta: (list::torch.tensor) List of additive FiLM parameters for conv layers (one for each channel).
        :return: (torch.tensor) Resulting representation after passing through layer.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self._film(out, gamma[0], beta[0])
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self._film(out, gamma[1], beta[1])

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def _film(self, x, gamma, beta):
        gamma = gamma[None, :, None, None]
        beta = beta[None, :, None, None]
        return gamma * x + beta


class ResNet(nn.Module):
    def __init__(self, block, layers, bn_fn, initial_pool=True, conv1_kernel_size=7):
        super(ResNet, self).__init__()
        self.initial_pool = initial_pool # False for 84x84
        self.inplanes = self.curr_planes = 64
        self.conv1 = nn.Conv2d(3, self.curr_planes, kernel_size=conv1_kernel_size, stride=2, padding=1, bias=False)
        self.bn1 = bn_fn(self.curr_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.inplanes, layers[0], bn_fn)
        self.layer2 = self._make_layer(block, self.inplanes * 2, layers[1], bn_fn, stride=2)
        self.layer3 = self._make_layer(block, self.inplanes * 4, layers[2], bn_fn, stride=2)
        self.layer4 = self._make_layer(block, self.inplanes * 8, layers[3], bn_fn, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, TaskNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, bn_fn, stride=1):
        downsample = None
        if stride != 1 or self.curr_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.curr_planes, planes * block.expansion, stride),
                bn_fn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.curr_planes, planes, bn_fn, stride, downsample))
        self.curr_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.curr_planes, planes, bn_fn))

        return nn.Sequential(*layers)

    def _flatten(self, x):
        sz = x.size()
        return x.view(-1, sz[-3], sz[-2], sz[-1]) if x.dim() >=5 else x

    def forward(self, x, param_dict=None):
        x = self._flatten(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    @property
    def output_size(self):
        return 512

class FilmResNet(ResNet):
    """
    Wrapper object around BasicBlockFilm that constructs a complete ResNet with FiLM layer adaptation. Inherits from
    ResNet object, and works with identical logic.
    """

    def __init__(self, block, layers, bn_fn, initial_pool=True, conv1_kernel_size=7):
        ResNet.__init__(self, block, layers, bn_fn, initial_pool, conv1_kernel_size)
        self.layers = layers

    def _get_adaptation_layer(self, generatable=False):
        if generatable:
            return FilmLayerGenerator
        else:
            return FilmLayer
    
    def _get_adaptation_config(self):
        param_dict = {
                'num_maps_per_layer' : [self.inplanes, self.inplanes * 2, self.inplanes * 4, self.inplanes * 8],
                'num_blocks_per_layer' : [len(self.layer1), len(self.layer2), len(self.layer3), len(self.layer4)]
                }
        return param_dict

    def forward(self, x, param_dict):
        """
        Forward pass through ResNet. Same logic as standard ResNet, but expects a dictionary of FiLM parameters to be
        provided (by adaptation network objects).
        :param x: (torch.tensor) Batch of images to pass through ResNet.
        :param param_dict: (list::dict::torch.tensor) One dictionary for each block in each layer of the ResNet,
                           containing the FiLM adaptation parameters for each conv layer in the model.
        :return: (torch.tensor) Feature representation after passing through adapted network.
        """
        x = self._flatten(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        for block in range(self.layers[0]):
            x = self.layer1[block](x, param_dict[0][block]['gamma'], param_dict[0][block]['beta'])
        for block in range(self.layers[1]):
            x = self.layer2[block](x, param_dict[1][block]['gamma'], param_dict[1][block]['beta'])
        for block in range(self.layers[2]):
            x = self.layer3[block](x, param_dict[2][block]['gamma'], param_dict[2][block]['beta'])
        for block in range(self.layers[3]):
            x = self.layer4[block](x, param_dict[3][block]['gamma'], param_dict[3][block]['beta'])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

def resnet18(pretrained=False, pretrained_model_path=None, batch_norm='basic', with_film=False, **kwargs):
    """
        Constructs a ResNet-18 model.
    """
    nl = get_normalisation_layer(batch_norm)
    if with_film:
        model = FilmResNet(BasicBlockFilm, [2, 2, 2, 2], nl, **kwargs)
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], nl, **kwargs)

    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path)
        model.load_state_dict(ckpt_dict['state_dict'])

    return model

def resnet18_84(pretrained=False, pretrained_model_path=None, batch_norm='basic', with_film=False, **kwargs):
    """
        Constructs a ResNet-18 model for 84 x 84 images.
    """
    nl = get_normalisation_layer(batch_norm)
    if with_film:
        model = FilmResNet(BasicBlockFilm, [2, 2, 2, 2], nl, initial_pool=False, conv1_kernel_size=5, **kwargs)
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], nl, initial_pool=False, conv1_kernel_size=5, **kwargs)

    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path)
        model.load_state_dict(ckpt_dict['state_dict'])

    return model
