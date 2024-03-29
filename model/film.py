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
from timm.models.efficientnet_blocks import ConvBnAct, InvertedResidual, CondConvResidual, EdgeResidual

def tag_film_layers(feature_extractor_name, feature_extractor):
    if 'efficientnet' in feature_extractor_name:
        def recursive_tag(module, name):
            if isinstance(module, EdgeResidual) or isinstance(module, ConvBnAct):
                modules_to_tag = ['bn1']
            elif isinstance(module, InvertedResidual) or isinstance(module, CondConvResidual):
                modules_to_tag = ['bn2']
            elif isinstance(module, EfficientNet): # tag batch norms in root
                modules_to_tag = ['bn1', 'bn2']
            else: 
                modules_to_tag = []
            for child_module_name in dir(module):
                child_module = getattr(module, child_module_name)
                child_module_type = type(child_module)
                if child_module_name in modules_to_tag and issubclass(child_module_type, nn.BatchNorm2d):
                    child_module.film = True
            for name, child in module.named_children():
                recursive_tag(child, name)
        recursive_tag(feature_extractor, 'feature_extractor')
    elif 'vit' in feature_extractor_name:
        def recursive_tag(module, name):
            for child_module_name in dir(module):
                child_module = getattr(module, child_module_name)
                child_module_type = type(child_module)
                if child_module_name in ['norm', 'norm1', 'norm2'] and (issubclass(child_module_type, nn.LayerNorm) or issubclass(child_module_type, nn.GroupNorm)):
                    child_module.film = True
            for name, child in module.named_children():
                recursive_tag(child, name)
        recursive_tag(feature_extractor, 'feature_extractor')

def get_film_parameter_names(feature_extractor_name, feature_extractor):
    parameter_list = []
    for name, module in feature_extractor.named_modules():
        if hasattr(module, 'film'):
            parameter_list.append(name + '.weight')
            parameter_list.append(name + '.bias')
    return parameter_list

def unfreeze_film(film_parameter_names, feature_extractor):
    for name, param in feature_extractor.named_parameters():
        if name in film_parameter_names:
            param.requires_grad = True

def get_film_parameters(film_parameter_names, feature_extractor):
    film_params = {}
    if not film_parameter_names == None:
        for name, param in feature_extractor.named_parameters():
            if name in film_parameter_names:
                film_params[name] = param.detach().clone()
    return film_params

def get_film_parameter_sizes(film_parameter_names, feature_extractor):
    film_params_sizes = {}
    for name, param in feature_extractor.named_parameters():
        if name in film_parameter_names:
            film_params_sizes[name] = len(param)
    return film_params_sizes
