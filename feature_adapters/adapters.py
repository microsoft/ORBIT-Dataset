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

import torch.nn as nn

class FilmAdapter(nn.Module):
    """
    Class for FiLM adapter network (outputs FiLM layer parameters for all layers in a base feature extractor). Used when --adapt_features is True (for when --feature_adaptation_method is 'generate' or 'learn').
    """
    def __init__(self, layer, adaptation_config, task_dim=None):
        """
        Creates instances of FilmAdapter.
        :param layer: (FilmLayer or FilmLayerGenerator) Objecet that produces FiLM layer parameters.
        :param adaptation_config: (dict::list::int or dict::list::list::int) Number of FiLM maps and blocks per layer in the base feature extractor.
        :param task_dim: (int or None) Dimensionality of task embedding.
        :return: Nothing.
        """
        super().__init__()
        self.num_maps = adaptation_config['num_maps_per_layer']
        self.num_blocks = adaptation_config['num_blocks_per_layer']
        self.task_dim = task_dim
        self.num_target_layers = len(self.num_maps)
        self.layer = layer
        self.layers = self.get_layers()

    def get_layers(self):
        """
        Function that creates FiLM layers (FilmLayer/FilmLayerGenerator) according to number of maps and blocks specified per layer in the base feature extractor.
        :return: (nn.ModuleList::self.layer) List of all the FiLM layers in the base feature extractor.
        """
        layers = nn.ModuleList()
        for num_maps, num_blocks in zip(self.num_maps, self.num_blocks):
            layers.append(
                self.layer(
                    num_maps=num_maps,
                    num_blocks=num_blocks,
                    task_dim=self.task_dim
                )
            )
        return layers

    def _init_layers(self):
        """
        Function that initialises the FiLM adapter network. Used only if the layer class is FilmLayer (not FilmLayerGenerator).
        :return: Nothing.
        """
        for layer in range(self.num_target_layers):
            self.layers[layer]._init_layer()
    
    def forward(self, x):
        """
        Function that performs a forward pass of a task embedding through the FiLM adapter network to get the FiLM layer parameters.
        :param x: (torch.Tensor) Task embedding. Note, x is not used if FilmLayer (only for FilmLayerGenerator).
        :return: (list::dict::torch.Tensor or list::dict::list::torch.Tensor) Parameters of all FiLM layers.
        """
        return [self.layers[layer](x) for layer in range(self.num_target_layers)]

    def regularization_term(self, switch_device=False):
        """
        Function that computes the regularisation term for the FiLM adapter network.
        :param switch_device: (bool) If True, move regularisation term to first GPU.
        :return: (torch.scalar) Regularisation term.
        """
        l2_term = 0
        for layer in self.layers:
            l2_term += layer.regularization_term()
        return l2_term.cuda(0) if switch_device else l2_term

class NullAdapter(nn.Module):
    """
    Class for a null adapter network when --adapt_features is False
    """
    def __init__(self):
        """
        Creates instances of NullAdapter.
        :return: Nothing.
        """
        super().__init__()

    def _init_layers(self):
        pass

    def forward(self, x):
        return {}

    def regularization_term(self, switch_device=False):
        return 0
