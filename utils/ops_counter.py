import timm
import numpy as np
import torch.nn as nn
from typing import List
from thop import profile
from thop import clever_format
from thop.profile import register_hooks
from thop.vision.basic_hooks import count_convNd

class OpsCounter():
    def __init__(self, count_backward=False):
        self.verbose = False # verbosity of thop package
        self.multiplier=2 if count_backward else 1 # counts foward + backward pass MACs
        self.task_mac_counter, self.task_params_counter = 0, 0
        self.personalise_time_per_task = []
        self.inference_time_per_frame = []
        self.custom_ops = None
        self.set_custom_ops()

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

        film_generator_params, set_encoder_params, film_params = 0, 0, 0
        if base_model.adapt_features:
            # film generator params
            if hasattr(base_model, 'film_generator'):
                for param in base_model.film_generator.parameters():
                    film_generator_params += param.numel()
            # set encoder params
            if hasattr(base_model, 'set_encoder'):
                for param in base_model.set_encoder.parameters():
                    set_encoder_params += param.numel()
            # film params
            film_params = sum(base_model.film_parameter_sizes.values())

        self.base_params_counter = feature_extractor_params + classifier_params + film_generator_params + set_encoder_params + film_params
        feature_extractor_params, classifier_params, film_generator_params, set_encoder_params, film_params = clever_format([feature_extractor_params, classifier_params, film_generator_params, set_encoder_params, film_params], "%.2f")
        self.params_break_down = "feature extractor: {0:}, classifier: {1:}, film generator: {2:}, set encoder: {3:}, film params {4:}".format(feature_extractor_params, classifier_params, film_generator_params, set_encoder_params, film_params)

    def add_macs(self, num_macs):
        self.task_mac_counter += num_macs

    def add_params(self, num_params):
        self.task_params_counter += num_params

    def get_uncounted_modules(self, model: nn.Module) -> List[str]:
        """
        Get list of modules that are not counted by default by thop, or have already have a custom op defined in self.custom_ops
        Note, this only considers leaf modules (which can also include modules with no parameters, e.g. NullSetEncoder())
        """
        uncounted_mods = []
        for m in model.modules():
            if len(list(m.children())) > 0: # skip non-leaf modules
                continue
            m_type = type(m)
            if m_type not in register_hooks and m_type not in self.custom_ops:
                uncounted_mods.append(str(m_type))

        return list(set(uncounted_mods))

    def set_custom_ops(self):
        """
        Set custom ops for modules that are not counted by default by thop
        """
        self.custom_ops = {
            timm.models.layers.Conv2dSame: count_convNd
        }

    def compute_macs(self, module, *inputs):
        list_inputs = []
        for input in inputs:
            list_inputs.append(input)
        macs, params = profile(module, inputs=inputs, custom_ops=self.custom_ops, verbose=self.verbose)
        self.add_macs(macs * self.multiplier)
        self.add_params(params)

    def task_complete(self):
        self.task_mac_counter = 0
        self.task_params_counter = 0

    def get_task_macs(self):
        return self.task_mac_counter

    def get_task_params(self):
        return self.base_params_counter + self.task_params_counter