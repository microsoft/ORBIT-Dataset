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
from abc import ABC, abstractmethod
from collections import OrderedDict

from model.mlps import DenseResidualBlock

class LinearClassifier(nn.Module):
    """
    Class for a linear classification layer.
    """
    def __init__(self, feat_dim, logit_scale: float=1.0):
        """
        Creates instance of LinearClassifier.
        :param feat_dim: (int) Size of feature extractor output.
        :param logit_scale: (float) Scale factor for logits.
        :return: Nothing.
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.logit_scale = logit_scale

    def init(self, num_classes:int):
        """
        Function that creates and initialises a linear classification layer.
        :param num_classes: (int) Number of classes in classification layer.
        :return: Nothing.
        """
        self.weight = nn.Parameter(torch.zeros(num_classes, self.feat_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

    def predict(self, features, ops_counter=None):
        """
        Function that passes a batch of target features through linear classification layer to get logits over object classes for each feature.
        :param features: (torch.Tensor) Batch of features.
        :return: (torch.Tensor) Logits over object classes for each feature.
        """
        num_samples = features.size(0)
        num_classes = self.weight.size(0)
        logits = self.logit_scale * F.linear(features, self.weight, self.bias)

        if ops_counter:
            ops_counter.add_macs(num_classes * num_samples * self.feat_dim)
        
        return logits

    def reset(self):
        self.weight = None
        self.bias = None

class HeadClassifier(nn.Module, ABC):
    """
    Class for a head-style classifier which is created by a computation of (context) features. Similar to https://github.com/cambridge-mlg/cnaps.
    """
    def __init__(self, logit_scale: float=1.0):
        """
        Creates instance of HeadClassifier.
        :param logit_scale: (float) Scale factor for logits.
        :return: Nothing.
        """
        super().__init__()
        self.logit_scale = logit_scale

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

    @abstractmethod
    def reset(self):
        pass

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    @staticmethod
    def _mean_pooling(x):
        return torch.mean(x, dim=0, keepdim=True)

class VersaClassifier(HeadClassifier):
    """
    Class for a Versa classifier (https://github.com/cambridge-mlg/cnaps). Context features are passed through two hyper-networks to generate the weight and bias parameters of a linear classification layer, respectively.
    """
    def __init__(self, in_size, logit_scale: float=1.0):
        """
        Creates instance of VersaClassifier.
        :param in_size: (int) Size of feature extractor output.
        :param logit_scale: (float) Scale factor for logits.
        :return: Nothing.
        """
        super().__init__(logit_scale)
        self.weight_processor = self._make_layer(in_size, in_size)
        self.bias_processor = self._make_layer(in_size, 1)
        self.reset()

    def reset(self):
        self.weight = None
        self.bias = None

    @staticmethod
    def _make_layer(in_size, out_size):
        return DenseResidualBlock(in_size, out_size)

    def predict(self, target_features):
        """
        Function that passes a batch of target features through linear classification layer to get logits over object classes for each feature.
        :param target_features: (torch.Tensor) Batch of target features.
        :return: (torch.Tensor) Logits over object classes for each target feature.
        """
        return self.logit_scale * F.linear(target_features, self.weight, self.bias)

    def configure(self, context_features, context_labels, ops_counter=None):
        """
        Function that passes per-class context features through two hyper-networks to generate the weight vector and bias scalar for each class in a linear classification layer.
        :param context_features: (torch.Tensor) Context features.
        :param context_labels: (torch.Tensor) Corresponding class labels for context features.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: Nothing.
        """
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

        self.weight = torch.nn.Parameter(torch.cat(class_weight, dim=0))
        self.bias = torch.nn.Parameter(torch.reshape(torch.cat(class_bias, dim=1), [num_classes, ]))

class PrototypicalClassifier(HeadClassifier):
    """
    Class for a ProtoNets classifier using Euclidean distance (https://github.com/jakesnell/prototypical-networks).
    """
    def __init__(self, logit_scale: float=1.0, distance_fn: str='euclidean'):
        """
        Creates instance of PrototypicalClassifier.
        :param logit_scale: (float) Scale factor for logits.
        :param distance_fn: (str) If 'euclidean', compute Euclidean distance, if 'cosine', compute cosine distance between target and class weight vectors. 
        :return: Nothing.
        """
        super().__init__(logit_scale)
        self.distance_fn = distance_fn
        self.reset()

    def reset(self):
        self.weight = None
        if self.distance_fn == 'euclidean':
            self.bias = None

    def predict(self, features, ops_counter=None):
        """
        Function that classifiers a batch of features to get logits over object classes for each feature.
        :param features: (torch.Tensor) Batch of features.
        :return: (torch.Tensor) Logits over object classes for each target feature.
        """
        num_samples, feat_dim = features.size()
        num_classes = self.weight.size(0)
        if self.weight is None or (self.distance_fn == 'euclidean' and self.bias is None):
            raise AttributeError("Weight and/or bias not set - is model personalised?")
        if self.distance_fn == 'euclidean':
            logits = self.logit_scale * F.linear(features, self.weight, self.bias)
        elif self.distance_fn == 'cosine':
            expanded_features = features.repeat(num_classes,1,1).permute(1,2,0)
            expanded_weight = self.weight.repeat(num_samples,1,1).permute(0,2,1)
            logits = self.logit_scale * (F.cosine_similarity(expanded_features, expanded_weight, dim=1))
        else:
            raise ValueError(f"Distance function {self.distance_fn} not valid.")
        
        if ops_counter:
            if self.distance_fn == 'euclidean':
                ops_counter.add_macs(num_samples * feat_dim * num_classes)
            elif self.distance_fn == 'cosine':
                ops_counter.add_macs(num_samples * feat_dim * num_classes) # numerator dot product
                ops_counter.add_macs(num_classes * feat_dim) # denom norm A
                ops_counter.add_macs(num_samples * feat_dim) # denom norm B
                ops_counter.add_macs(num_samples * feat_dim * num_classes) # numerator/denomenator

        return logits

    def configure(self, context_features, context_labels, ops_counter=None):
        """
        Function that computes the per-class mean of context features and sets this as the class weight vector in a linear classification layer.
        :param context_features: (torch.Tensor) Context features.
        :param context_labels: (torch.Tensor) Corresponding class labels for context features.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: Nothing.
        """
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
            if self.distance_fn == 'euclidean':
                class_bias.append((-torch.matmul(nu, nu.t()))[None, None])
            if ops_counter:
                ops_counter.add_macs(nu.size(0) * nu.size(1)) # 2* in class weight
                ops_counter.add_macs(nu.size(0)**2 * nu.size(1)) # matmul in  class bias
                ops_counter.add_macs(nu.size(0) * nu.size(1)) # -1* in  class bias

        self.weight = torch.nn.Parameter(torch.cat(class_weight, dim=0))
        if self.distance_fn == 'euclidean':
            self.bias = torch.nn.Parameter(torch.reshape(torch.cat(class_bias, dim=1), [num_classes, ]))

class MahalanobisClassifier(HeadClassifier):
    """
    Class for a Mahalanobis classifier (https://github.com/peymanbateni/simple-cnaps). Computes per-class distributions using context features. Target features are classified by the shortest Mahalanobis distance to these distributions.
    """
    def __init__(self, logit_scale: float=1.0):
        """
        Creates instance of MahalanobisClassifier.
        :param logit_scale: (float) Scale factor for logits.
        :return: Nothing.
        """
        super().__init__(logit_scale)
        self.reset()

    def reset(self):
        self.means = None
        self.precisions = None
        self.task_mean = None
        self.task_precision = None

    def configure(self, context_features, context_labels, ops_counter=None):
        """
        Function that computes a per-class distribution (mean, precision) using the context features.
        :param context_features: (torch.Tensor) Context features.
        :param context_labels: (torch.Tensor) Corresponding class labels for context features.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: Nothing.
        """
        assert context_features.size(0) == context_labels.size(0), "context features and labels are different sizes!"

        means = []
        precisions = []
        task_covariance_estimate = self._estimate_cov(context_features, ops_counter)
        task_precision = torch.inverse(task_covariance_estimate + torch.eye(context_features.size(1), context_features.size(1)).to(context_features.device))
        task_mean = torch.mean(context_features, dim=0)

        label_set, _ = torch.sort(torch.unique(context_labels))

        for c in label_set:
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

        self.means = torch.nn.Parameter((torch.stack(means)))
        self.task_mean = torch.nn.Parameter(task_mean)
        self.precisions = torch.nn.Parameter((torch.stack(precisions)))
        self.task_precision = torch.nn.Parameter(task_precision)

    def predict(self, target_features):
        """
        Function that processes a batch of target features to get logits over object classes for each feature. Target features are classified by their Mahalanobis distance to the class means including the class precisions.
        :param target_features: (torch.Tensor) Batch of target features.
        :return: (torch.Tensor) Logits over object classes for each target feature.
        """
        # grabbing the number of classes and query examples for easier use later in the function
        number_of_classes = self.means.size(0)
        number_of_targets = target_features.size(0)

        # calculate the Mahalanobis distance between target features and the class means including the class precision
        repeated_target = target_features.repeat(1, number_of_classes).view(-1, self.means.size(1))
        repeated_class_means = self.means.repeat(number_of_targets, 1)
        repeated_difference = (repeated_class_means - repeated_target)
        repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                       repeated_difference.size(1)).permute(1, 0, 2)
        first_half = torch.matmul(repeated_difference, self.precisions)
        logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

        return self.logit_scale * logits

    @staticmethod
    def _estimate_cov(examples, ops_counter=None):
        """
        Estimate a covariance matrix given data.
        :param examples: (torch.Tensor) A 1-D or 2-D array containing multiple variables and observations. Each row of `m` represents a variable, and each column a single observation of all those variables.
        :return: (torch.Tensor) Covariance matrix of the variables.
        """
        if examples.size(0) > 1:
            cov_matrix = torch.cov(examples.t(), correction=1)
        else:
            factor = 1.0 / (examples.size(1) - 1)
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
            cov_matrix = factor * examples.matmul(examples.t()).squeeze()

        if ops_counter:
            ops_counter.add_macs(examples.size(0) * examples.size(1)) # computing mean
            ops_counter.add_macs(examples.size(0)**2 * examples.size(1)) # computing matmul
            ops_counter.add_macs(examples.size(0) * examples.size(1)) # computing factor*cov_matrix

        return cov_matrix
