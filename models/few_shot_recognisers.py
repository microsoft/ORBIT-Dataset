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

from features import extractors
from feature_adapters import FilmAdapter, NullAdapter
from models.poolers import MeanPooler
from models.set_encoder import SetEncoder, NullSetEncoder
from models.classifiers import VersaClassifier, PrototypicalClassifier, MahalanobisClassifier

class FewShotRecogniser(nn.Module):
    """
        Generic few-shot classification model
        :param args: (argparse.NameSpace) command line arguments
        :param with_task_encoder: (bool) specifies whether model should include a task encoder hypernetwork. If True, feature_adapter generates FiLM parameters, otherwise FiLM parameters are simply added/learned as additional model parameters. Only used with adapt_features == True
    """
    def __init__(self, args):
        super(FewShotRecogniser, self).__init__()
        self.args = args
        pretrained=True if self.args.pretrained_extractor_path else False
        
        # configure feature extractor
        extractor_fn = extractors[ self.args.feature_extractor ]
        self.feature_extractor = extractor_fn(
            pretrained=pretrained,
            pretrained_model_path=self.args.pretrained_extractor_path,
            batch_norm=self.args.batch_normalisation,
            with_film=self.args.adapt_features
        )
        
        # configure feature adapter
        if self.args.adapt_features:     
            if self.args.feature_adaptation_method == 'generate':
                self.set_encoder = SetEncoder(self.args.batch_normalisation)
                adaptation_layer = self.feature_extractor._get_adaptation_layer(generatable=True)
            else:
                self.set_encoder = NullSetEncoder()
                adaptation_layer = self.feature_extractor._get_adaptation_layer(generatable=False)
            self.feature_adapter = FilmAdapter(
                layer=adaptation_layer,
                adaptation_config = self.feature_extractor._get_adaptation_config(),
                task_dim=self.set_encoder.output_size
            ) 
        else:
            self.set_encoder = NullSetEncoder()
            self.feature_adapter = NullAdapter() 
         
        # configure classifier head
        if self.args.classifier == 'none': 
            self.classifier = None # classifier head will instead be appended per-task during train/test
        elif self.args.classifier == 'versa':
            self.classifier = VersaClassifier(self.feature_extractor.output_size)
        elif self.args.classifier == 'proto':
            self.classifier = PrototypicalClassifier()
        elif self.args.classifier == 'mahalanobis':
            self.classifier = MahalanobisClassifier() 
            
        # configure frame pooler
        self.frame_pooler = MeanPooler(T=self.args.clip_length)
       
        if not self.args.learn_extractor:
            self._freeze_extractor() # freeze the parameters of the feature extractor
    
    def _distribute_model(self):
        self.feature_extractor.cuda(1)
        self.feature_adapter.cuda(1)
    
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
 
    def _pool_features(self, features):

        return self.frame_pooler(features) # pool over frames per clip
    
    def _freeze_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def _set_batch_norm_mode(self, context):
        """
        Controls the batch norm mode in the feature extractor.
        :param context: Set to true when processing the context set and False when processing the target set.
        :return: Nothing
        """
        if self.args.batch_normalisation == "basic":
            self.feature_extractor.eval()  # always in eval mode
        else:
            # "task_norm-i" - respect context flag, regardless of state
            if context:
                self.feature_extractor.train()  # use train when processing the context set
            else:
                self.feature_extractor.eval()  # use eval when processing the target set
    
    def _compute_personalise_ops(self, ops_counter, context_frames, context_features):
        ops_counter.compute_macs(self.set_encoder, context_frames)
        ops_counter.compute_macs(self.feature_adapter, self.task_representation.cuda(1) if self.args.use_two_gpus else self.task_representation)
        ops_counter.compute_macs(self.feature_extractor, context_frames.cuda(1) if self.args.use_two_gpus else context_frames, self.feature_adapter_params)
        ops_counter.add_macs(context_features.size(0) * context_features.size(1) * self.args.clip_length) # MACs in _pool_features


class MultiStepFewShotRecogniser(FewShotRecogniser):
    """
    Few-shot recogniser class that is personalised in multiple forward-backward steps (e.g. MAML, FineTuner). Each forward() call is one step.
    """
    def __init__(self, args):
        FewShotRecogniser.__init__(self, args)
       
    def forward(self, frames, ops_counter=None, test_mode=False):
        
        self.task_representation = self.set_encoder(frames)
        self.feature_adapter_params = self.feature_adapter(self.task_representation.cuda(1) if self.task_representation and self.args.use_two_gpus else self.task_representation)

        features = self._get_features(frames, context=not test_mode)
        features = self._pool_features(features)
        logits = self.classifier(features)

        if ops_counter:
            ops_counter.compute_macs(self.classifier, features)
            self._compute_personalise_ops(ops_counter, frames, features)
        
        return logits
    
    def _init_task_specific_params(self, num_classes, device, init_zeros=True):
        # add classifier to model
        self._init_classifier(num_classes, device, init_zeros)

        # add feature adapter to model
        if self.args.adapt_features:
            self.feature_adapter._init_layers()
            self.feature_adapter.to(device)

    def _init_classifier(self, way, device, init_zeros=True):
        self.classifier = nn.Linear(self.feature_extractor.output_size, way)
        if init_zeros:
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        else:
            nn.init.kaiming_uniform_(self.classifier.weight, mode="fan_out")
            nn.init.zeros_(self.classifier.bias)
        self.classifier.to(device)
  
    def _reset(self):
        self.classifier = None

class SingleStepFewShotRecogniser(FewShotRecogniser):
    """
    Few-shot recogniser class that is personalised in a single forward step (e.g. CNAPs, ProtoNets). Each personalise() call adapts the model to the task's context set. Each forward() call makes predictions on the task's target set. 
    """
    def __init__(self, args):
        FewShotRecogniser.__init__(self, args)

    def personalise(self, context_set, context_labels, ops_counter=None):
       
        self.task_representation = self.set_encoder(context_set)
        self.feature_adapter_params = self.feature_adapter(self.task_representation.cuda(1) if self.task_representation and self.args.use_two_gpus else self.task_representation)
        
        context_features = self._get_features(context_set, context=True)
        context_features = self._pool_features(context_features)
        
        self.classifier.configure(context_features, context_labels, ops_counter)
        
        if ops_counter:
            self._compute_personalise_ops(ops_counter, context_set, context_features)
            ops_counter.task_complete()

    def forward(self, target_set, test_mode=False):

        target_features = self._get_features(target_set)
        target_features = self._pool_features(target_features)
        
        return self.classifier.predict(target_features)

    def _reset(self):
        self.classifier.reset() 
