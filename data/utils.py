# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
import torch.nn as nn

def attach_frame_history(frames, history_length): 
    """
    Function to attach the immediate history of history_length frames to each frame in a tensor of frame data.
    param frames: (torch.Tensor) Frames.
    :param history_length: (int) Number of frames of history to append to each frame.
    :return: (torch.Tensor) Frames with attached frame history.
    """
    # pad with first frame so that frames 0 to history_length-1 can be evaluated
    frame_0 = frames.narrow(0, 0, 1)
    frames = torch.cat((frame_0.repeat(history_length-1, 1, 1, 1), frames), dim=0)

    # for each frame, attach its immediate history of history_length frames
    frames = [ frames ]
    for l in range(1, history_length):
        frames.append( frames[0].roll(shifts=-l, dims=0) )
    frames_with_history = torch.stack(frames, dim=1) # of size num_clips x history_length
    
    if history_length > 1:
        return frames_with_history[:-(history_length-1)] # frames have wrapped around, remove last (history_length - 1) frames
    else:
        return frames_with_history

def unpack_task(task_dict, device, context_to_device=True, target_to_device=False):
   
    context_clips = task_dict['context_clips']
    context_paths = task_dict['context_paths']
    context_labels = task_dict['context_labels']
    context_annotations = task_dict['context_annotations']
    target_clips = task_dict['target_clips']
    target_paths = task_dict['target_paths']
    target_labels = task_dict['target_labels']
    target_annotations = task_dict['target_annotations']
    object_list = task_dict['object_list']

    if context_to_device and isinstance(context_labels, torch.Tensor):
        context_labels = context_labels.to(device)
    if target_to_device and isinstance(target_labels, torch.Tensor):
        target_labels = target_labels.to(device)
  
    return context_clips, context_paths, context_labels, target_clips, target_paths, target_labels, object_list

def get_batch_indices(index, last_element, batch_size):
        batch_start_index = index * batch_size
        batch_end_index = batch_start_index + batch_size
        if batch_end_index > last_element:
            batch_end_index = last_element
        return batch_start_index, batch_end_index
