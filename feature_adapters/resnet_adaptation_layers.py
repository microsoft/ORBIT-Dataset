"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file adaptation_networks.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/adaptation_networks.py)
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
import torch.nn.functional as F

from models.mlps import DenseBlock

class BaseFilmLayer(nn.Module):
    def __init__(self, num_maps, num_blocks):
        super(BaseFilmLayer, self).__init__()
        
        self.num_maps = num_maps
        self.num_blocks = num_blocks
        self.num_layers_per_block = 2
    
    def regularization_term(self):
        l2_term = 0
        for block_gamma_regularizers, block_beta_regularizers in zip(self.gamma_regularizers, self.beta_regularizers):
            for gamma_regularizer, beta_regularizer in zip(block_gamma_regularizers, block_beta_regularizers):
                l2_term += (gamma_regularizer ** 2).sum()
                l2_term += (beta_regularizer ** 2).sum()
        return l2_term

class FilmLayer(BaseFilmLayer):
    def __init__(self, num_maps, num_blocks, task_dim=None):
        BaseFilmLayer.__init__(self, num_maps, num_blocks)

        self._init_layer()

    def _init_layer(self):
        self.gammas, self.gamma_regularizers = nn.ModuleList(), nn.ModuleList()
        self.betas, self.beta_regularizers = nn.ModuleList(), nn.ModuleList()

        for block in range(self.num_blocks):
            gammas, gamma_regularizers = nn.ParameterList(), nn.ParameterList()
            betas, beta_regularizers = nn.ParameterList(), nn.ParameterList()
            for layer in range(self.num_layers_per_block):
                gammas.append(nn.Parameter(torch.ones(self.num_maps), requires_grad=True))
                gamma_regularizers.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(self.num_maps), 0, 0.001), requires_grad=True))

                betas.append(nn.Parameter(torch.zeros(self.num_maps), requires_grad=True))
                beta_regularizers.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(self.num_maps), 0, 0.001), requires_grad=True))
            self.gammas.append(gammas)
            self.gamma_regularizers.append(gamma_regularizers)
            self.betas.append(betas)
            self.beta_regularizers.append(beta_regularizers)
    
    def forward(self, x):
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = { 'gamma' : [], 'beta' : []}
            for layer in range(self.num_layers_per_block):
                block_param_dict['gamma'].append(self.gammas[block][layer] * self.gamma_regularizers[block][layer] + torch.ones_like(self.gamma_regularizers[block][layer]))
                block_param_dict['beta'].append(self.betas[block][layer] * self.beta_regularizers[block][layer])
            block_params.append(block_param_dict)

        return block_params

class FilmLayerGenerator(BaseFilmLayer):
    def __init__(self, num_maps, num_blocks, task_dim):
        BaseFilmLayer.__init__(self, num_maps, num_blocks)
        
        self.task_dim = task_dim

        self.gamma_generators, self.gamma_regularizers = nn.ModuleList(), nn.ModuleList()
        self.beta_generators, self.beta_regularizers = nn.ModuleList(), nn.ModuleList()

        for block in range(self.num_blocks):
            gamma_generators, gamma_regularizers = nn.ModuleList(), nn.ParameterList()
            beta_generators, beta_regularizers = nn.ModuleList(), nn.ParameterList()
            for layer in range(self.num_layers_per_block):
                gamma_generators.append(self._make_layer(self.task_dim, self.num_maps))
                gamma_regularizers.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(self.num_maps), 0, 0.001), requires_grad=True))

                beta_generators.append(self._make_layer(self.task_dim, self.num_maps))
                beta_regularizers.append(torch.nn.Parameter(torch.nn.init.normal_(torch.empty(self.num_maps), 0, 0.001), requires_grad=True))
            
            self.gamma_generators.append(gamma_generators)
            self.gamma_regularizers.append(gamma_regularizers)
            self.beta_generators.append(beta_generators)
            self.beta_regularizers.append(beta_regularizers)

    def _make_layer(self, in_size, out_size):
        return DenseBlock(in_size, out_size)
    
    def forward(self, x):
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = { 'gamma' : [], 'beta' : []}
            for layer in range(self.num_layers_per_block):
                block_param_dict['gamma'].append( self.gamma_generators[block][layer](x).squeeze() * self.gamma_regularizers[block][layer] + torch.ones_like(self.gamma_regularizers[block][layer]) )
                block_param_dict['beta'].append( self.beta_generators[block][layer](x).squeeze() * self.beta_regularizers[block][layer] )
            
            block_params.append(block_param_dict)
        
        return block_params
