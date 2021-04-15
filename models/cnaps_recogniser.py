"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file model.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/model.py) and
config_networks.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/config_networks.py)
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

from models import BaseRecogniser
from features import extractors
from models.set_encoder import SetEncoder
from feature_adapters import FilmAdapter, NullAdapter
from models.classifiers import linear_classifier, LinearClassifierAdapter, PrototypicalClassifierAdapter
from models.poolers import MeanPooler

class CNAPSRecogniser(BaseRecogniser):
    def __init__(self, args):
        BaseRecogniser.__init__(self, args)
        self._config_model(args)
        
        self.task_representation = None
        self.class_representations = OrderedDict()

    def personalise(self, context_set, context_labels, ops_counter=None):
        context_frames, context_paths, context_video_ids = context_set
        
        self.task_representation = self.set_encoder(context_frames)
        self.feature_adapter_params = self.feature_adapter(self.task_representation.cuda(1) if self.args.use_two_gpus else self.task_representation)
        
        context_features = self._get_features(context_frames, context=True)
        context_features = self._pool_features(context_features)
        
        # get the classifier parameters from the head-hypernet.
        self._build_class_reps(context_features, context_labels, ops_counter)
        self.classifier_params = self._get_classifier_params()
        
        if ops_counter:
            if self.args.adapt_features:
                ops_counter.compute_macs(self.set_encoder, context_frames)
                ops_counter.compute_macs(self.feature_adapter, self.task_representation.cuda(1) if self.args.use_two_gpus else self.task_representation)
                ops_counter.add_params(self.feature_adapter.num_generated_params)
            ops_counter.compute_macs(self.feature_extractor, context_frames.cuda(1) if self.args.use_two_gpus else context_frames, self.feature_adapter_params)
            ops_counter.add_macs(context_features.size(0) * context_features.size(1) * self.args.clip_length) # MACs in _pool_features
            ops_counter.compute_macs(self.classifier_adapter, self.class_representations)
            ops_counter.task_complete()

    def forward(self, target_set, test_mode=False):

        target_frames, target_paths, target_video_ids = target_set 

        target_features = self._get_features(target_frames)
        target_features = self._pool_features(target_features, test_mode)

        return self.classifier(target_features, self.classifier_params)

    def _get_features(self, frames, context=False):

        if self.args.use_two_gpus:
            frames_1 = frames.cuda(1)
            self._set_batch_norm_mode(context)
            features_1 = self.feature_extractor(frames_1, self.feature_adapter_params)
            features = features_1.cuda(0)
        else:
            self._set_batch_norm_mode(context)
            features = self.feature_extractor(frames, self.feature_adapter_params)

        return features
 
    def _build_class_reps(self, context_features, context_labels, ops_counter):
        """
        Return a Versa-style head by using class conditional inference
        :param context_features: features to make inference on
        :param context_labels: class association of videos
        :return: dictionary (output of head hypernet)
        """
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(context_features, 0, self._extract_class_indices(context_labels, c))
            class_rep = torch.mean(class_features, dim=0, keepdim=True)
            self.class_representations[c.item()] = class_rep
            if ops_counter:
                ops_counter.add_macs(class_features.size(0) * class_features.size(1)) # counts MACs in average
                ops_counter.add_params(class_rep.numel()) # parameters in class classifier weight
    
    def _get_classifier_params(self):
        classifier_params = self.classifier_adapter(self.class_representations)
        return classifier_params

    def _extract_class_indices(self, labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    def _distribute_model(self):
        self.feature_extractor.cuda(1)
        self.feature_adapter.cuda(1)

    def _config_model(self, args):

        pretrained=True if args.pretrained_extractor_path else False
        self.set_encoder = SetEncoder(args.batch_normalisation)
        task_dim = self.set_encoder.pre_pooling_fn.output_size

        if args.adapt_features:
            extractor_fn = extractors[ 'film_' + args.feature_extractor ]
            self.feature_extractor = extractor_fn(
                pretrained=pretrained,
                pretrained_model_path=args.pretrained_extractor_path,
                batch_norm=args.batch_normalisation
            )
            adaptation_layer = self.feature_extractor._get_adaptation_layer(generator=True)
            adaptation_config = self.feature_extractor._get_adaptation_config()
            self.feature_adapter = FilmAdapter(
                layer=adaptation_layer,
                adaptation_config = adaptation_config,
                task_dim=task_dim
            )
        else:
            extractor_fn = extractors[ args.feature_extractor ]
            self.feature_extractor = extractor_fn(
                pretrained=pretrained,
                pretrained_model_path=args.pretrained_extractor_path,
                batch_norm=args.batch_normalisation
            )
            self.feature_adapter = NullAdapter() 

        if args.classifier == 'versa':
            self.classifier_adapter = LinearClassifierAdapter(self.feature_extractor.output_size)
        elif args.classifier == 'proto':
            self.classifier_adapter = PrototypicalClassifierAdapter()
        self.classifier = linear_classifier
        
        self.frame_pooler = MeanPooler(T=args.clip_length)
       
        if not args.learn_extractor:
            # Freeze the parameters of the feature extractor
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
