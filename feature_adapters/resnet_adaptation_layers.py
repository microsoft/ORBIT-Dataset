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
    """
    Base class for a FiLM layer in a ResNet feature extractor. Will be wrapped around a FilmAdapter instance.
    """
    def __init__(self, num_maps, num_blocks):
        """
        Creates a BaseFilmLayer instance.
        :param num_maps: (int) Dimensionality of input to each block in the FiLM layer.
        :param num_blocks: (int) Number of blocks in the FiLM layer.
        :return: Nothing.
        """
        super(BaseFilmLayer, self).__init__()
        
        self.num_maps = num_maps
        self.num_blocks = num_blocks
        self.num_layers_per_block = 2
    
    def regularization_term(self):
        """
        Function that computes the L2-norm regularisation term for the FiLM layer. Recall, FiLM applies gamma * x + beta. As such, params gamma and beta are regularized to unity, i.e. ||gamma - 1||_2 and ||beta||_2.
        :return: (torch.scalar) L2-norm regularisation term.
        """
        l2_term = 0
        for block_gamma_regularizers, block_beta_regularizers in zip(self.gamma_regularizers, self.beta_regularizers):
            for gamma_regularizer, beta_regularizer in zip(block_gamma_regularizers, block_beta_regularizers):
                l2_term += (gamma_regularizer ** 2).sum()
                l2_term += (beta_regularizer ** 2).sum()
        return l2_term

class FilmLayer(BaseFilmLayer):
    """
    Class for a learnable FiLM layer in an EfficientNet feature extractor. Here, the FiLM layer is a set of nn.ModuleList() made up of nn.ParameterList()s made up of nn.Parameter()s, which are updated via standard gradient steps.
    """
    def __init__(self, num_maps, num_blocks, task_dim=None):
        """
        Creates a FilmLayer instance.
        :param num_maps: (int) Dimensionality of input to each block in the FiLM layer.
        :param num_blocks: (int) Number of blocks in the FiLM layer.
        :param task_dim: (None) Not used.
        :return: Nothing.
        """
        BaseFilmLayer.__init__(self, num_maps, num_blocks)
        self._init_layer()

    def _init_layer(self):
        """
        Function that creates and initialises the FiLM layer. The FiLM layer has a nn.ModuleList() for its gammas and betas (and their corresponding regularisers). Each element in a nn.ModuleList() is a nn.ParamaterList() and correspondings to one block in the FiLM layer. Each element in a nn.ParameterList() is a nn.Parameter() of size self.num_maps and corresponds to a layer in the block.
        :return: Nothing.
        """
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
        """
        Function that returns the FiLM layer's parameters. Note, input x is ignored.
        :param x: (None) Not used.
        :return: (list::dict::list::nn.Parameter) Parameters of the FiLM layer.
        """
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = { 'gamma' : [], 'beta' : []}
            for layer in range(self.num_layers_per_block):
                block_param_dict['gamma'].append(self.gammas[block][layer] * self.gamma_regularizers[block][layer] + torch.ones_like(self.gamma_regularizers[block][layer]))
                block_param_dict['beta'].append(self.betas[block][layer] * self.beta_regularizers[block][layer])
            block_params.append(block_param_dict)

        return block_params

class FilmLayerGenerator(BaseFilmLayer):
    """
    Class for a generated FiLM layer in a ResNet feature extractor. Here, the task embedding is passed through two hyper-networks which generate the FiLM layer's parameters.
    """
    def __init__(self, num_maps, num_blocks, task_dim):
        """
        Creates a FilmLayerGenerator instance.
        :param num_maps: (int) Dimensionality of input to each block in the FiLM layer.
        :param num_blocks: (int) Number of blocks in the FiLM layer.
        :param task_dim: (None) Dimensionality of task embedding.
        :return: Nothing.
        """
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
        """
        Function that performs a forward pass of a task embedding through the generators to get the FiLM layer's parameters.
        :param x: (torch.Tensor) Task embedding.
        :return: (list::dict::list::torch.Tensor) Parameters of the FiLM layer.
        """
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = { 'gamma' : [], 'beta' : []}
            for layer in range(self.num_layers_per_block):
                block_param_dict['gamma'].append( self.gamma_generators[block][layer](x).squeeze() * self.gamma_regularizers[block][layer] + torch.ones_like(self.gamma_regularizers[block][layer]) )
                block_param_dict['beta'].append( self.beta_generators[block][layer](x).squeeze() * self.beta_regularizers[block][layer] )
            
            block_params.append(block_param_dict)
        
        return block_params
