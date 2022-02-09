import numpy as np
import torch.nn as nn
from thop import profile
from thop import clever_format

class OpsCounter():
    def __init__(self, count_backward=False):
        self.verbose = False
        self.multiplier=2 if count_backward else 1 # counts foward + backward pass MACs 
        self.task_mac_counter, self.task_params_counter, self.task_time = 0, 0, 0
        self.macs, self.params, self.time = [], [], []

    def set_base_params(self, base_model):
        
        # feature extractor params
        feature_extractor_params = 0
        for param in base_model.feature_extractor.parameters():
            feature_extractor_params += param.numel()

        # classifier params
        classifier_params = 0
        if isinstance(base_model.classifier, nn.Module):
            for param in base_model.classifier.parameters():
                classifier_params += param.numel()

        feature_adapter_params, set_encoder_params = 0, 0
        if base_model.adapt_features:
            # feature adapter params
            for param in base_model.feature_adapter.parameters():
                feature_adapter_params += param.numel()
            # set encoder params
            for param in base_model.set_encoder.parameters():
                set_encoder_params += param.numel()

        self.base_params_counter = feature_extractor_params + classifier_params + feature_adapter_params + set_encoder_params
        feature_extractor_params, classifier_params, feature_adapter_params, set_encoder_params = clever_format([feature_extractor_params, classifier_params, feature_adapter_params, set_encoder_params], "%.2f")
        self.params_break_down = "feature extractor: {0:}, classifier: {1:}, feature adapter: {2:}, set encoder: {3:}".format(feature_extractor_params, classifier_params, feature_adapter_params, set_encoder_params)

    def add_macs(self, num_macs):
        self.task_mac_counter += num_macs

    def add_params(self, num_params):
        self.task_params_counter += num_params

    def log_time(self, time):
        self.task_time += time

    def compute_macs(self, module, *inputs):
        list_inputs = []
        for input in inputs:
            list_inputs.append(input)
        custom_ops = module.thop_custom_ops if hasattr(module, 'thop_custom_ops') else {}
        macs, params = profile(module, inputs=inputs, custom_ops=custom_ops, verbose=self.verbose)
        self.add_macs(macs * self.multiplier)

    def task_complete(self):
        self.macs.append(self.task_mac_counter)
        self.params.append(self.base_params_counter + self.task_params_counter)
        self.time.append(self.task_time)
        self.task_mac_counter = 0
        self.task_params_counter = 0
        self.task_time = 0

    def get_macs(self):
        return clever_format([self.macs[-1]], "%.2f")

    def get_mean_stats(self):
        mean_ops = np.mean(self.macs)
        std_ops = np.std(self.macs)
        mean_params = np.mean(self.params)
        mean_ops, std_ops, mean_params = clever_format([mean_ops, std_ops, mean_params], "%.2f")
        mean_time = np.mean(self.time)
        std_time = np.std(self.time)
        return "MACs to personalise: {0:} ({1:}) time to personalise: {2:.2f}s ({3:.2f}s) #learnable params {4:} ({5:})".format(mean_ops, std_ops, mean_time, std_time, mean_params, self.params_break_down)
