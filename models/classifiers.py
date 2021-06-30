"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file adaptation_networks.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/adaptation_networks.py)
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
from collections import OrderedDict

from models.mlps import DenseResidualBlock

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _build_class_reps(self, context_features, context_labels, ops_counter):
        class_reps = OrderedDict()
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(context_features, 0, self._extract_class_indices(context_labels, c))
            class_rep = self._mean_pooling(class_features)
            class_reps[c.item()] = class_rep
            if ops_counter:
                ops_counter.add_macs(context_features.size(0)) # selecting class features
                ops_counter.add_macs(class_features.size(0) * class_features.size(1)) # mean pooling

        return class_reps

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector
    
    @staticmethod
    def _mean_pooling(x):
        return torch.mean(x, dim=0, keepdim=True)
 
class VersaClassifier(Classifier):
    def __init__(self, d_theta):
        super().__init__()
        self.weight_processor = self._make_layer(d_theta, d_theta)
        self.bias_processor = self._make_layer(d_theta, 1)
        self.param_dict = {}

    @staticmethod
    def _make_layer(in_size, out_size):
        return DenseResidualBlock(in_size, out_size)
    
    def predict(self, target_features):
        return F.linear(target_features, self.param_dict['weight'], self.param_dict['bias'])

    def configure(self, context_features, context_labels, ops_counter):
        assert context_features.size(0) == context_labels.size(0), "context features and labels are different sizes!" 
        class_rep_dict = self._build_class_reps(context_features, context_labels, ops_counter)
        class_weight = []
        class_bias = []

        label_set = list(class_rep_dict.keys())
        label_set.sort()
        num_classes = len(label_set)

        for class_num in label_set:
            nu = class_rep_dict[class_num]
            class_weight.append(self.weight_processor(nu))
            class_bias.append(self.bias_processor(nu))
            if ops_counter:
                ops_counter.compute_macs(self.weight_processor, nu)
                ops_counter.compute_macs(self.bias_processor, nu)

        self.param_dict['weight'] = torch.cat(class_weight, dim=0)
        self.param_dict['bias'] = torch.reshape(torch.cat(class_bias, dim=1), [num_classes, ])

    def reset(self):
        self.param_dict = {}

class PrototypicalClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.param_dict = {}

    def predict(self, target_features):
        return F.linear(target_features, self.param_dict['weight'], self.param_dict['bias'])

    def configure(self, context_features, context_labels, ops_counter):
        assert context_features.size(0) == context_labels.size(0), "context features and labels are different sizes!" 
        class_rep_dict = self._build_class_reps(context_features, context_labels, ops_counter)
        class_weight = []
        class_bias = []

        label_set = list(class_rep_dict.keys())
        label_set.sort()
        num_classes = len(label_set)

        for class_num in label_set:
            # equation 8 from the prototypical networks paper
            nu = class_rep_dict[class_num]
            class_weight.append(2 * nu)
            class_bias.append((-torch.matmul(nu, nu.t()))[None, None])
            if ops_counter:
                ops_counter.add_macs(nu.size(0) * nu.size(1)) # 2* in class weight
                ops_counter.add_macs(nu.size(0)**2 * nu.size(1)) # matmul in  class bias
                ops_counter.add_macs(nu.size(0) * nu.size(1)) # -1* in  class bias

        self.param_dict['weight'] = torch.cat(class_weight, dim=0)
        self.param_dict['bias'] = torch.reshape(torch.cat(class_bias, dim=1), [num_classes, ])
        
    def reset(self):
        self.param_dict = {}

class MahalanobisClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.param_dict = {}

    def configure(self, context_features, context_labels, ops_counter):
        assert context_features.size(0) == context_labels.size(0), "context features and labels are different sizes!" 

        means = []
        precisions = []
        task_covariance_estimate = self._estimate_cov(context_features, ops_counter)
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(context_features, 0, self._extract_class_indices(context_labels, c))
            # mean pooling examples to form class means
            means.append(self._mean_pooling(class_features).squeeze())
            lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
            class_covariance_estimate = self._estimate_cov(class_features, ops_counter)
            covariance_matrix = (lambda_k_tau * class_covariance_estimate) \
                                + ((1 - lambda_k_tau) * task_covariance_estimate) \
                                + torch.eye(class_features.size(1), device=class_features.device)
            precisions.append(torch.inverse(covariance_matrix))

            if ops_counter:
                ops_counter.add_macs(context_features.size(0)) # selecting class features
                ops_counter.add_macs(class_features.size(0) * class_features.size(1)) # mean pooling
                ops_counter.add_macs(1) # computing lambda_k_tau
                ops_counter.add_macs(class_covariance_estimate.size(0) * class_covariance_estimate.size(1)) # lambda_k_tau * class_covariance_estimate
                ops_counter.add_macs(task_covariance_estimate.size(0) * task_covariance_estimate.size(1)) # (1-lambda_k_tau) * task_covariance_estimate
                ops_counter.add_macs(1/3*covariance_matrix.size(0) ** 3 + covariance_matrix.size(0) ** 2 - 4/3*covariance_matrix.size(0)) # computing inverse of covariance_matrix, taken from https://en.wikipedia.org/wiki/Gaussian_elimination#Computational_efficiency
                # note, sum of 3 matrices to compute covariance_matrix is not included here

        self.param_dict['means'] = (torch.stack(means))
        self.param_dict['precisions'] = (torch.stack(precisions))
         
    def predict(self, target_features):
        # grabbing the number of classes and query examples for easier use later in the function
        number_of_classes = self.param_dict['means'].size(0)
        number_of_targets = target_features.size(0)

        """
        Calculating the Mahalanobis distance between query examples and the class means
        including the class precision estimates in the calculations, reshaping the distances
        and multiplying by -1 to produce the sample logits
        """
        repeated_target = target_features.repeat(1, number_of_classes).view(-1, self.param_dict['means'].size(1))
        repeated_class_means = self.param_dict['means'].repeat(number_of_targets, 1)
        repeated_difference = (repeated_class_means - repeated_target)
        repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                       repeated_difference.size(1)).permute(1, 0, 2)
        first_half = torch.matmul(repeated_difference, self.param_dict['precisions'])
        logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

        return logits

    def reset(self):
        self.param_dict = {}

    @staticmethod
    def _estimate_cov(examples, ops_counter, rowvar=False, inplace=False):
        """
        SCM: unction based on the suggested implementation of Modar Tensai
        and his answer as noted in: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5
        Estimate a covariance matrix given data.
        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.
        Returns:
            The covariance matrix of the variables.
        """
        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        cov_matrix = factor * examples.matmul(examples_t)

        if ops_counter:
            ops_counter.add_macs(examples.size(0) * examples.size(1)) # computing mean
            ops_counter.add_macs(1) # computing factor
            ops_counter.add_macs(examples.size(0)**2 * examples.size(1)) # computing matmul
            ops_counter.add_macs(examples.size(0) * examples.size(1)) # computing factor*cov_matrix

        return cov_matrix
