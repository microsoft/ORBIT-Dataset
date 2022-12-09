"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file set_encoder.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/set_encoder.py)
from the cambridge-mlg/cnaps library (https://github.com/cambridge-mlg/cnaps).

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

from model.feature_extractors import create_feature_extractor

class SetEncoder(nn.Module):
    """
    Simple set encoder implementing DeepSets (https://arxiv.org/abs/1703.06114). Used for modeling permutation-invariant representations on sets (mainly for extracting task-level embedding of context sets).
    """
    def __init__(self, encoder_name='efficientnet_b0', task_embedding_dim=64):
        """
        Creates an instance of SetEncoder.
        :return: Nothing.
        """
        super(SetEncoder, self).__init__()
        self.task_embedding_dim = task_embedding_dim
        self.encoder_name = encoder_name

        self.encoder, _ = create_feature_extractor(
                                    feature_extractor_name=self.encoder_name, 
                                    pretrained=True,
                                    with_film=False,
                                    learn_extractor=False
                                    )
        self.reducer = nn.Linear(self.encoder.output_size, self.task_embedding_dim)                                
 
    def forward(self, x):
        """
        Function that encodes a set of N elements into N embeddings, each of dim self.task_embedding_dim
        :param x: (torch.Tensor) Set of elements (for clips it has the shape: batch x clip length x C x H x W).
        :return: (torch.Tensor) Individual element embeddings.
        """
        with torch.no_grad():
            x = self._flatten(x)
            x = self.encoder(x)
        return self.reducer(x)

    def _flatten(self, x):
        if x.dim() == 5:
            return x.flatten(end_dim=1)
        else:
            return x

    def aggregate(self, x, reduction='mean'):
        """
        Function that aggregates the encoded elements in x.
        :param x: (torch.Tensor) Set of encoded elements (i.e. from forward()).
        :param reduction: (str) If 'mean', average the encoded elements in x, otherwise do not average.
        :return: (torch.Tensor) Mean representation of the set as a single vector if reduction = 'mean', otherwise as a set of encoded elements.
        """
        x = torch.cat(x, dim=0)
        if reduction == 'mean':
            x = torch.mean(x, dim=0, keepdim=True)
        return x

    @property
    def output_size(self):
        return self.task_embedding_dim

class NullSetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return None

    def aggregate(self, x, reduction='mean'):
        return None
    
    @property
    def output_size(self):
        return None
