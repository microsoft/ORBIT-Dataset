# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

class MeanPooler(nn.Module):
    def __init__(self, T, dim=1):
        super(MeanPooler, self).__init__()
        self.T = T
        self.dim=dim

    def forward(self, x):
        feat_dim = x.size(-1)
        x = x.view(-1, self.T, feat_dim)
        return torch.mean(x, dim=self.dim)
    
class IdentityPooler(nn.Module):
    def __init__(self):
        super(IdentityPooler, self).__init__()
        pass

    def forward(self, x):
        return x
