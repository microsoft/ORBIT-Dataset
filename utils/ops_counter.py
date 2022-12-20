import numpy as np
import torch.nn as nn
from thop import profile
from thop import clever_format

class OpsCounter():
    def __init__(self, count_backward=False):
        self.verbose = False
        self.multiplier=2 if count_backward else 1 # counts foward + backward pass MACs 
        self.task_mac_counter, self.task_params_counter = 0, 0
        self.macs, self.params = [], []
        self.personalise_time_per_task = []
        self.inference_time_per_frame = []

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
            film_params = sum(base_model.film_parameter_sizes)

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
        self.task_mac_counter = 0
        self.task_params_counter = 0

    def get_macs(self):
        return clever_format([self.macs[-1]], "%.2f")

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

    def get_mean_stats(self):
        mean_ops = np.mean(self.macs)
        std_ops = np.std(self.macs)
        mean_params = np.mean(self.params)
        mean_ops, std_ops, mean_params = clever_format([mean_ops, std_ops, mean_params], "%.2f")
        mean_personalise_time = self.convert_to_minutes(np.mean(self.personalise_time_per_task))
        std_personalise_time = self.convert_to_minutes(np.std(self.personalise_time_per_task))
        mean_inference_time = self.convert_to_microseconds(np.mean(self.inference_time_per_frame))
        std_inference_time = self.convert_to_microseconds(np.std(self.inference_time_per_frame))
        return f"MACs to personalise: {mean_ops} ({std_ops}) time to personalise: {mean_personalise_time} ({std_personalise_time}) inference time per frame: {mean_inference_time} ({std_inference_time}) #params {mean_params} ({self.params_break_down})"
