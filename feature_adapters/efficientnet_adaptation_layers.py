# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

from models.mlps import DenseBlock

class BaseFilmLayer(nn.Module):
    def __init__(self, num_maps, num_blocks):
        super(BaseFilmLayer, self).__init__()

        self.num_maps = num_maps
        self.num_blocks = num_blocks

    def regularization_term(self):
        """
        Compute the regularization term for the parameters. Recall, FiLM applies gamma * x + beta. As such, params
        gamma and beta are regularized to unity, i.e. ||gamma - 1||_2 and ||beta||_2.
        :return: (torch.tensor) Scalar for l2 norm for all parameters according to regularization scheme.
        """
        l2_term = 0
        for gamma_regularizer, beta_regularizer in zip(self.gamma_regularizers, self.beta_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        return l2_term

class FilmLayer(BaseFilmLayer):
    def __init__(self, num_maps, num_blocks, task_dim=None):
        BaseFilmLayer.__init__(self, num_maps, num_blocks)

        self._init_layer()

    def _init_layer(self):

        self.gammas, self.gamma_regularizers = nn.ParameterList(), nn.ParameterList()
        self.betas, self.beta_regularizers = nn.ParameterList(), nn.ParameterList() 
        
        for i in range(self.num_blocks):
            self.gammas.append(nn.Parameter(torch.ones(self.num_maps[i]), requires_grad=True))
            self.gamma_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(self.num_maps[i]), 0, 0.001), requires_grad=True))
            self.betas.append(nn.Parameter(torch.zeros(self.num_maps[i]), requires_grad=True))
            self.beta_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(self.num_maps[i]), 0, 0.001), requires_grad=True))

    def forward(self, x):
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma': self.gammas[block] * self.gamma_regularizers[block] + torch.ones_like(self.gamma_regularizers[block]),
                'beta': self.betas[block] * self.beta_regularizers[block]
                            }
            block_params.append(block_param_dict)
        return block_params

class FilmLayerGenerator(BaseFilmLayer):
    def __init__(self, num_maps, num_blocks, task_dim):
        BaseFilmLayer.__init__(self, num_maps, num_blocks)
        self.task_dim = task_dim

        self.gamma_generators, self.gamma_regularizers = nn.ModuleList(), nn.ParameterList()
        self.beta_generators, self.beta_regularizers = nn.ModuleList(), nn.ParameterList()

        for i in range(self.num_blocks):
            self.gamma_generators.append(self._make_layer(self.task_dim, num_maps[i]))
            self.gamma_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(num_maps[i]), 0, 0.001),
                                                               requires_grad=True))

            self.beta_generators.append(self._make_layer(self.task_dim, num_maps[i]))
            self.beta_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(num_maps[i]), 0, 0.001),
                                                              requires_grad=True))

    def _make_layer(self, in_size, out_size):
        return DenseBlock(in_size, out_size)

    def forward(self, x):
        """
        Forward pass through adaptation network.
        :param x: (torch.tensor) Input representation to network (task level representation z).
        :return: (list::dictionaries) Dictionary for every block in layer. Dictionary contains all the parameters
                 necessary to adapt layer in base network. Base network is aware of dict structure and can pull params
                 out during forward pass.
        """
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma': self.gamma_generators[block](x).squeeze() * self.gamma_regularizers[block] +
                          torch.ones_like(self.gamma_regularizers[block]),
                'beta': self.beta_generators[block](x).squeeze() * self.beta_regularizers[block],
            }
            block_params.append(block_param_dict)
        return block_params
