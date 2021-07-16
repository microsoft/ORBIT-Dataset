# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
import torch.nn as nn
import data.transforms as dt

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

class FrameLoader(nn.Module):
    """
    Class to load frames (organised by clip) from disk into a tensor.
    """
    def __init__(self, clip_length, frame_size):
        super().__init__()
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.transformation = dt.frame_transform

    def forward(self, frame_paths, device):
        """ 
        Function to load frames from disk and return them as a tensor of num_clips x self.clip_length x 3 x self.frame_size x self.frame_size.
        :param frame_paths: (np.ndarray) Frame paths organised as clips of self.clip_length frames.
        :param device: (torch.device) Device to load frames to.
        :return: (torch.Tensor) Tensor of transformed frame data corresponding to frame_paths.
    """ 
        num_clips = len(frame_paths)
        frames = torch.zeros(num_clips, self.clip_length, 3, self.frame_size, self.frame_size, device=device)
        for c in range(num_clips):
            for f in range(self.clip_length):
                frames[c, f] = self.transformation(frame_paths[c,f])
 
        return frames

def attach_frame_history(frame_paths, labels, clip_length):
    """
    Function to attach the immediate history of clip_length frames to each frame in a tensor of frames.
    :param frame_paths: (np.ndarray) Frame paths.
    :param labels: (torch.Tensor) Object labels.
    :param clip_length: (int) Number of frames of history to append to each frame.
    :return: (np.ndarray, torch.Tensor) Frame paths with attached frame history and replicated object labels.
    """
    # expand labels
    labels = labels.view(-1,1).repeat(1, clip_length).view(-1)
    
    # pad with first frame so that frames 0 to clip_length-1 can be evaluated
    frame_paths = frame_paths.reshape(-1)
    frame_paths = np.concatenate([np.repeat(frame_paths[0], clip_length-1), frame_paths])
    
    # for each frame path, attach its immediate history of clip_length frames
    frame_paths = [ frame_paths ]
    for l in range(1, clip_length):
        frame_paths.append( np.roll(frame_paths[0], shift=-l, axis=0) )
    frame_paths_with_history = np.stack(frame_paths, axis=1) # of size num_clips x clip_length
    
    # since frame_paths_with_history have wrapped around, remove last (clip_length - 1) frames
    return frame_paths_with_history[:-(clip_length-1)], labels 

def unpack_task(task_dict, device):
    context_clips = task_dict['context_clips']
    context_labels = task_dict['context_labels']
    target_clips = task_dict['target_clips']
    target_labels = task_dict['target_labels']

    # only send context and target labels to device; all others will be sent only when processed
    context_labels = context_labels.to(device)
    target_labels = target_labels.to(device) if isinstance(target_labels, torch.Tensor) else target_labels
    return context_clips, context_labels, target_clips, target_labels
