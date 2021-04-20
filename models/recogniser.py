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

import torch.nn as nn

from models import BaseRecogniser
from features import extractors
from feature_adapters import FilmAdapter, NullAdapter 
from models.poolers import MeanPooler

class Recogniser(BaseRecogniser):
    def __init__(self, args):
        BaseRecogniser.__init__(self, args)
        self._config_model(args)
        
        self.null_input = None

    def forward(self, x, ops_counter=None, test_mode=False):

        frames, paths, video_ids = x
        features = self._get_features(frames, not test_mode)
        features = self._pool_features(features, test_mode)
        logits = self.classifier(features)

        if ops_counter:
            ops_counter.compute_macs(self.feature_extractor, frames, self.feature_adapter_params)
            ops_counter.add_macs(features.size(0) * features.size(1) * self.args.clip_length) # MACs in pool_features
            ops_counter.compute_macs(self.classifier, features)

        return logits

    def _init_classifier(self, way, init_zeros=False, ops_counter=None):
        self.classifier = nn.Linear(self.feature_extractor.output_size, way)
        if init_zeros:
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        else:
            nn.init.kaiming_uniform_(self.classifier.weight, mode="fan_out")
            nn.init.zeros_(self.classifier.bias)
        
        if ops_counter:
            ops_counter.add_params(self.classifier.weight.numel() + self.classifier.bias.numel())

    def _init_film_layers(self):
        self.feature_adapter._init_layers()

    def _get_features(self, x, context=False):

        self._set_batch_norm_mode(context)
        self.feature_adapter_params = self.feature_adapter(self.null_input) 
        features = self.feature_extractor(x, self.feature_adapter_params)

        return features
    
    def _config_model(self, args):
        pretrained=True if args.pretrained_extractor_path else False

        if args.adapt_features:
            extractor_fn = extractors[ 'film_' + args.feature_extractor ]
            self.feature_extractor = extractor_fn(
                pretrained=pretrained,
                pretrained_model_path=args.pretrained_extractor_path,
                batch_norm=args.batch_normalisation
            )
            adaptation_layer = self.feature_extractor._get_adaptation_layer()
            adaptation_config = self.feature_extractor._get_adaptation_config()
            self.feature_adapter = FilmAdapter(
                layer=adaptation_layer,
                adaptation_config=adaptation_config
            )
        else:
            extractor_fn = extractors[ args.feature_extractor ]
            self.feature_extractor = extractor_fn(
                pretrained=pretrained,
                pretrained_model_path=args.pretrained_extractor_path,
                batch_norm=args.batch_normalisation
            )
            self.feature_adapter = NullAdapter() 
   
        self.frame_pooler = MeanPooler(T=args.clip_length)
    
        self.classifier = None

        if not args.learn_extractor:
            # freeze the parameters of feature extractor
            for param in self.feature_extractor.parameters():
                param.requires_grad = False 
