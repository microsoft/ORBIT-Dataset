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
        self.initial_film_parameters = initial_film_parameters
        self.film_parameter_names = list(self.initial_film_parameters.keys())
        self.film_parameter_names.sort()
        self.generators = nn.ModuleList()
        self.regularizers = nn.ParameterList()

        for i, film_param_name in enumerate(self.film_parameter_names):
            self.generators.append(self._make_generator(pooled_size, hidden_size, film_parameter_sizes[film_param_name]))
            self.regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(film_parameter_sizes[film_param_name]), 0, 0.001),
                                                  requires_grad=True))

            self.l2_term = 0.0

    def _apply(self, fn): # ensures self.initial_film_parameters is moved to device
        super(FilmParameterGenerator, self)._apply(fn)
        self.initial_film_parameters = { k: fn(v) for k,v in self.initial_film_parameters.items()}
        return self

    def _make_generator(self, pooled_size, hidden_size, out_size):
        return DenseBlock(pooled_size, hidden_size, out_size)

    def regularization_term(self):
        return self.l2_term

    def forward(self, x):
        film_dict = {}
        self.l2_term = 0.0
        for i, film_param_name in enumerate(self.film_parameter_names):
            if 'weight' in film_param_name:
                generated_weight = self.generators[i](x).squeeze() * self.regularizers[i] + torch.ones_like(self.regularizers[i])
                new_film_param = self.initial_film_parameters[film_param_name] * generated_weight
            elif 'bias' in film_param_name:
                generated_bias = self.generators[i](x).squeeze() * self.regularizers[i]
                new_film_param = self.initial_film_parameters[film_param_name] + generated_bias
            self.l2_term += (self.regularizers[i] ** 2).sum()  # not exactly the same as weight decay as we are not taking the square root
            film_dict[film_param_name] = new_film_param
        return film_dict
    
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
