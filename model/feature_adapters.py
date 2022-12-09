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

from model.mlps import DenseBlock

class FilmParameterGenerator(nn.Module):
    """
    Class for generating FiLM parameters for a base feature extractor. Used when --adapt_features is True.
    """
    def __init__(self, film_parameter_sizes, initial_film_parameters, pooled_size, hidden_size):
        super().__init__()
        self.num_film_layers = len(film_parameter_sizes)
        self.initial_film_parameters = initial_film_parameters
        self.generators = nn.ModuleList()
        self.regularizers = nn.ParameterList()
        for i in range(self.num_film_layers):
            self.generators.append(self._make_generator(pooled_size, hidden_size, film_parameter_sizes[i]))
            self.regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(film_parameter_sizes[i]), 0, 0.001),
                                                  requires_grad=True))

            self.l2_term = 0.0

    def _make_generator(self, pooled_size, hidden_size, out_size):
        return DenseBlock(pooled_size, hidden_size, out_size)

    def regularization_term(self):
        return self.l2_term

    def forward(self, x):
        film_parameters = []
        self.l2_term = 0.0
        for i in range(self.num_film_layers):
            generated_values = self.generators[i](x).squeeze()
            self.l2_term += (generated_values ** 2).sum()  # not exactly the same as weight decay as we are not taking the square root
            film_parameters.append(generated_values + self.initial_film_parameters[i])
        return film_parameters
    
class NullGenerator(nn.Module):
    """
    Class for a null film generator network when --adapt_features is False
    """
    def __init__(self):
        """
        Creates instances of NullGenerator.
        :return: Nothing.
        """
        super().__init__()

    def forward(self, x):
        return {}

    def regularization_term(self):
        return 0
