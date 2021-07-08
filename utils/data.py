import torch
import numpy as np

class ListBatcher():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def _get_number_of_batches(self, list_size):
        self._set_list_size(list_size)
        assert self.batch_size > 0, "self.batch_size is 0" 
        assert self.list_size > 0, "self.list_size is 0"
        return int(np.ceil(float(self.list_size) / float(self.batch_size)))

    def _set_list_size(self, list_size):
        self.list_size = list_size

    def _get_batch_indices(self, index):
        batch_start_index = index * self.batch_size
        batch_end_index = batch_start_index + self.batch_size
        if batch_end_index > self.list_size:
            batch_end_index = self.list_size
        return range(batch_start_index, batch_end_index)

    def reset(self, batch_size):
        self.batch_size = batch_size
        self.list_size = 0

def select_batch(list_of_tensors, batch_range=None, device=None):
    selected_batches = []
    for tensor in list_of_tensors:
        if device:
            selected_batch = tensor[batch_range].to(device)
        else:
            selected_batch = tensor[batch_range]
        selected_batches.append(selected_batch)

    return tuple(selected_batches) if len(selected_batches)>1 else selected_batches[0]

def send_to_device(task_set, task_labels, device):
    task_set = task_set.to(device)
    task_labels = task_labels.to(device)
    return task_set, task_labels
    
def unpack_task(task_dict, device, test_mode=False):
    context_set, context_labels = send_to_device(task_dict['context_set'], task_dict['context_labels'], device)

    if test_mode: # test_mode; group target set by videos
        target_set_by_video, target_labels_by_video = task_dict['target_set'], task_dict['target_labels']
        return context_set, context_labels, target_set_by_video, target_labels_by_video
    else:
        target_set, target_labels = send_to_device(task_dict['target_set'], task_dict['target_labels'], device)
        return context_set, context_labels, target_set, target_labels 
