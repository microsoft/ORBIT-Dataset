# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

from models.mlps import DenseBlock

class BaseFilmLayer(nn.Module):
    """
    Base class for a FiLM layer in an EfficientNet feature extractor. Will be wrapped around a FilmAdapter instance.
    """
    def __init__(self, num_maps, num_blocks):
        """
        Creates a BaseFilmLayer instance.
        :param num_maps: (list::int) Dimensionality of input to each block in the FiLM layer.
        :param num_blocks: (int) Number of blocks in the FiLM layer.
        :return: Nothing.
        """
        super(BaseFilmLayer, self).__init__()

        self.num_maps = num_maps
        self.num_blocks = num_blocks

    def regularization_term(self):
        """
        Function that computes the L2-norm regularisation term for the FiLM layer. Recall, FiLM applies gamma * x + beta. As such, params gamma and beta are regularized to unity, i.e. ||gamma - 1||_2 and ||beta||_2.
        :return: (torch.scalar) L2-norm regularisation term.
        """
        l2_term = 0
        for gamma_regularizer, beta_regularizer in zip(self.gamma_regularizers, self.beta_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        return l2_term

class FilmLayer(BaseFilmLayer):
    """
    Class for a learnable FiLM layer in an EfficientNet feature extractor. Here, the FiLM layer is a set of nn.ParameterList()s made up of nn.Parameter()s, which are updated via standard gradient steps.
    """
    def __init__(self, num_maps, num_blocks, task_dim=None):
        """
        Creates a FilmLayer instance.
        :param num_maps: (list::int) Dimensionality of input to each block in the FiLM layer.
        :param num_blocks: (int) Number of blocks in the FiLM layer.
        :param task_dim: (None) Not used.
        :return: Nothing.
        """
        BaseFilmLayer.__init__(self, num_maps, num_blocks)
        self._init_layer()

    def _init_layer(self):
        """
        Function that creates and initialises the FiLM layer. The FiLM layer has a nn.ParameterList() for its gammas and betas (and their corresponding regularisers). Each element in a nn.ParamaterList() is a nn.Parameter() of size self.num_maps and corresponds to one block in the FiLM layer.
        :return: Nothing.
        """
        self.gammas, self.gamma_regularizers = nn.ParameterList(), nn.ParameterList()
        self.betas, self.beta_regularizers = nn.ParameterList(), nn.ParameterList() 
        
        for i in range(self.num_blocks):
            self.gammas.append(nn.Parameter(torch.ones(self.num_maps[i]), requires_grad=True))
            self.gamma_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(self.num_maps[i]), 0, 0.001), requires_grad=True))
            self.betas.append(nn.Parameter(torch.zeros(self.num_maps[i]), requires_grad=True))
            self.beta_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(self.num_maps[i]), 0, 0.001), requires_grad=True))

    def forward(self, x):
        """
        Function that returns the FiLM layer's parameters. Note, input x is ignored.
        :param x: (None) Not used.
        :return: (list::dict::nn.Parameter) Parameters of the FiLM layer.
        """
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma': self.gammas[block] * self.gamma_regularizers[block] + torch.ones_like(self.gamma_regularizers[block]),
                'beta': self.betas[block] * self.beta_regularizers[block]
                            }
            block_params.append(block_param_dict)
        return block_params

class FilmLayerGenerator(BaseFilmLayer):
    """
    Class for a generated FiLM layer in an EfficientNet feature extractor. Here, the task embedding is passed through two hyper-networks which generate the FiLM layer's parameters.
    """
    def __init__(self, num_maps, num_blocks, task_dim):
        """
        Creates a FilmLayerGenerator instance.
        :param num_maps: (list::int) Dimensionality of input to each block in the FiLM layer.
        :param num_blocks: (int) Number of blocks in the FiLM layer.
        :param task_dim: (None) Dimensionality of task embedding.
        :return: Nothing.
        """
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
        Function that performs a forward pass of a task embedding through the generators to get the FiLM layer's parameters.
        :param x: (torch.Tensor) Task embedding.
        :return: (list::dict::torch.Tensor) Parameters of the FiLM layer.
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
