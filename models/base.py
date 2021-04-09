# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

class BaseRecogniser(nn.Module):
    def __init__(self, args):
        super(BaseRecogniser, self).__init__()
        self.args = args 
 
    def _pool_features(self, features, test_mode=False):

        if test_mode: # clip = history of self.args.clip_length frames for every frame
            features = self._attach_frame_history(features)

        # pool over frames per clip
        return self.frame_pooler(features)
    
    def _attach_frame_history(self, features):
        
        # pad with first feature so that first frames 0 to self.args.clip_length-1 can be evaluated
        features_0 = features.narrow(0, 0, 1)
        features = torch.cat((features_0.repeat(self.args.clip_length-1, 1), features), dim=0)
        
        # for each frame, attach its immediate history of self.args.clip_length frames
        features = [ features ]
        for l in range(1, self.args.clip_length):
            features.append( features[0].roll(shifts=-l, dims=0) )
        features = torch.stack(features, dim=1)
            
        # since frames have wrapped around, remove last (num_frames - 1) frames
        return features[:-(self.args.clip_length-1)]
    
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
