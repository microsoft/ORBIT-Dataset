import numpy as np
from thop import profile
from thop import clever_format

class OpsCounter():
    def __init__(self, count_backward=False):
        self.verbose = False
        self.multiplier=2 if count_backward else 1 # counts foward + backward pass MACs 
        self.task_mac_counter, self.task_params_counter = 0, 0
        self.macs, self.params = [], []

    def set_base_params(self, base_model):
        self.base_params_counter = 0
        for param in base_model.parameters():
            self.base_params_counter += param.numel()

    def add_macs(self, num_macs):
        self.task_mac_counter += num_macs

    def add_params(self, num_params):
        self.task_params_counter += num_params

    def compute_macs(self, module, *inputs):
        list_inputs = []
        for input in inputs:
            list_inputs.append(input)
        macs, params = profile(module, inputs=inputs, verbose=self.verbose)
        self.add_macs(macs * self.multiplier)

    def task_complete(self):
        self.macs.append(self.task_mac_counter)
        self.params.append(self.base_params_counter + self.task_params_counter)
        self.task_mac_counter = 0
        self.task_params_counter = 0

    def get_macs(self):
        return clever_format([self.macs[-1]], "%.3f")

    def get_mean_stats(self):
        mean_ops = np.mean(self.macs)
        std_ops = np.std(self.macs)
        mean_params = np.mean(self.params)
        std_params = np.std(self.params)
        mean_ops, std_ops, mean_params, std_params = clever_format([mean_ops, std_ops, mean_params, std_params], "%.3f")
        return "MACs: {0:} ({1:}) #params {2:} ({3:})".format(mean_ops, std_ops, mean_params, std_params)
