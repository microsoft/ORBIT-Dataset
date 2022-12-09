"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file film.py (https://github.com/cambridge-mlg/dp-fsl/blob/main/src/film.py).

The original license is included below:

MIT License

Copyright (c) 2022 John F. Bronskill

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
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.efficientnet import EfficientNet
from timm.models.layers.norm_act import BatchNormAct2d
from timm.models.efficientnet_blocks import ConvBnAct, InvertedResidual, CondConvResidual, EdgeResidual

def insert_film_layers(feature_extractor_name, feature_extractor):
    if 'efficientnet' in feature_extractor_name:
        def recursive_replace(module, name):
            if isinstance(module, EdgeResidual) or isinstance(module, ConvBnAct):
                modules_to_replace = ['bn1']
            elif isinstance(module, InvertedResidual) or isinstance(module, CondConvResidual):
                modules_to_replace = ['bn2']
            elif isinstance(module, EfficientNet): # replace batch norms in root
                modules_to_replace = ['bn1', 'bn2']
            else: 
                modules_to_replace = []
            for child_module_name in dir(module):
                child_module = getattr(module, child_module_name)
                if child_module_name in modules_to_replace and isinstance(child_module, BatchNormAct2d):
                    setattr(module, child_module_name, BatchNormAct2dFiLM(child_module.num_features))
            for name, child in module.named_children():
                recursive_replace(child, name)
        recursive_replace(feature_extractor, 'feature_extractor')

def get_film_parameter_names(feature_extractor_name, feature_extractor):
    parameter_list = []
    for name, module in feature_extractor.named_modules():
        if ('efficientnet' in feature_extractor_name and isinstance(module, BatchNormAct2dFiLM)) or \
                ('vit' in feature_extractor_name and isinstance(module, nn.LayerNorm)):
            parameter_list.append(name + '.weight')
            parameter_list.append(name + '.bias')
    return parameter_list

def enable_film(film_parameter_names, feature_extractor):
    for name, param in feature_extractor.named_parameters():
        if name in film_parameter_names:
            param.requires_grad = True

def get_film_parameters(film_parameter_names, feature_extractor):
    film_params = []
    if not film_parameter_names == None:
        for name, param in feature_extractor.named_parameters():
            if name in film_parameter_names:
                film_params.append(param.detach().clone())
    return film_params

def init_film_parameters(film_parameter_names, feature_extractor, device):
    if not film_parameter_names == None:
        for name, module in feature_extractor.named_modules():
            if name in film_parameter_names:
                if isinstance(module, BatchNormAct2dFiLM): # efficientnets
                    num_features = module.num_features
                    module.film_gamma = nn.Parameter(torch.ones(num_features), requires_grad=True).to(self.device)
                    module.film_beta = nn.Parameter(torch.zeros(num_features), requires_grad=True).to(self.device)
                elif isinstance(module, nn.LayerNorm): # vits
                    module.weight = nn.Parameter(torch.ones(num_features), requires_grad=True).to(self.device)
                    module.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True).to(self.device)

def get_film_parameter_sizes(film_parameter_names, feature_extractor):
    film_params_sizes = []
    for name, param in feature_extractor.named_parameters():
        if name in film_parameter_names:
            film_params_sizes.append(len(param))
    return film_params_sizes

def film_to_dict(film_parameter_names, film_parameters):
    film_dict = {}
    if not film_parameter_names == None:
        assert len(film_parameter_names) == len(film_parameters)
        for i in range(len(film_parameter_names)):
            film_dict[film_parameter_names[i]] = film_parameters[i]
    return film_dict

class BatchNormAct2dFiLM(BatchNormAct2d):
    """BatchNorm + Activation
    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 apply_act=True, act_layer=nn.ReLU, inplace=True, drop_layer=None):
        BatchNormAct2d.__init__(self, num_features, eps=eps, momentum=momentum, affine=affine,
                                track_running_stats=track_running_stats, apply_act=apply_act,
                                act_layer=act_layer, inplace=inplace, drop_layer=drop_layer)
        # initialize FiLM weights
        self.film_gamma = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.film_beta = nn.Parameter(torch.zeros(num_features), requires_grad=True)

    def forward(self, x):
        # cut & paste of torch.nn.BatchNorm2d.forward impl to avoid issues with torchscript and tracing
        assert x.ndim == 4, f'expected 4D input (got {x.ndim}D input)'

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

        x = film(x, self.film_gamma, self.film_beta)

        x = self.drop(x)
        x = self.act(x)
        return x

def film(x, gamma, beta):
    gamma = gamma[None, :, None, None]
    beta = beta[None, :, None, None]
    return gamma * x + beta
