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

    def log_time(self, time:float , time_type:str='personalise'):
        if time_type == 'personalise':
            self.personalise_time_per_task.append(time)
        elif time_type == 'inference':
            self.inference_time_per_frame.append(time)
        else:
            raise ValueError(f"time_type must be 'personalise' or 'inference' but got {time_type}")

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

    def get_macs(self):
        return self.task_mac_counter

    def get_params(self):
        return self.base_params_counter + self.task_params_counter

    def convert_to_minutes(self, seconds):
        mins, secs = divmod(seconds, 60)
        mins = round(mins)
        secs = round(secs)
        if mins == 0 and secs == 0:
            return f"{seconds:.2f}s"
        else:
            return f"{mins:d}m{secs:d}s"

    def convert_to_microseconds(self, seconds):
        return f"{round(seconds * 1000000):d}\u03bcs"

    def get_mean_stats(self, macs, params):
        mean_ops = np.mean(macs)
        std_ops = np.std(macs)
        mean_params = np.mean(params)
        mean_ops, std_ops, mean_params = clever_format([mean_ops, std_ops, mean_params], "%.2f")
        mean_personalise_time = self.convert_to_minutes(np.mean(self.personalise_time_per_task))
        std_personalise_time = self.convert_to_minutes(np.std(self.personalise_time_per_task))
        mean_inference_time = self.convert_to_microseconds(np.mean(self.inference_time_per_frame))
        std_inference_time = self.convert_to_microseconds(np.std(self.inference_time_per_frame))
        return f"MACs to personalise: {mean_ops} ({std_ops}) time to personalise: {mean_personalise_time} ({std_personalise_time}) inference time per frame: {mean_inference_time} ({std_inference_time}) #params {mean_params} ({self.params_break_down})"
