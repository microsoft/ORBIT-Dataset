# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tv_F

class DatasetFromClipPaths(Dataset):
    def __init__(self, clip_paths, with_labels):
        super().__init__()
        #TODO currently doesn't support loading of annotations
        self.with_labels = with_labels
        if self.with_labels:
            self.clip_paths, self.clip_labels = clip_paths
        else:
            self.clip_paths = clip_paths
        
        self.normalize_stats = {'mean' : [0.500, 0.436, 0.396], 'std' : [0.145, 0.143, 0.138]} # orbit mean train frame
        
    def __getitem__(self, index):
        clip = []
        for frame_path in self.clip_paths[index]:
            frame = self.load_and_transform_frame(frame_path)
            clip.append(frame)
    
        if self.with_labels:
            return torch.stack(clip, dim=0), self.clip_labels[index]
        else:
            return torch.stack(clip, dim=0)
    
    def load_and_transform_frame(self, frame_path):
        """
        Function to load and transform frame.
        :param frame_path: (str) Path to frame.
        :return: (torch.Tensor) Loaded and transformed frame.
        """
        frame = Image.open(frame_path)
        frame = tv_F.to_tensor(frame)
        return tv_F.normalize(frame, mean=self.normalize_stats['mean'], std=self.normalize_stats['std'])

    def __len__(self):
        return len(self.clip_paths)

def get_clip_loader(clips, batch_size, with_labels=False):
    if isinstance(clips[0], np.ndarray):
        clips_dataset = DatasetFromClipPaths(clips, with_labels=with_labels)
        return DataLoader(clips_dataset,
                      batch_size=batch_size,
                      num_workers=8,
                      pin_memory=True,
                      prefetch_factor=8,
                      persistent_workers=True)

    elif isinstance(clips[0], torch.Tensor):
        if with_labels:
            return list(zip(clips[0].split(batch_size), clips[1].split(batch_size)))
        else: 
            return clips.split(batch_size)

def attach_frame_history(frames, history_length):
    
    if isinstance(frames, np.ndarray):
        return attach_frame_history_paths(frames, history_length)
    elif isinstance(frames, torch.Tensor):
        return attach_frame_history_tensor(frames, history_length)

def attach_frame_history_paths(frame_paths, history_length):
    """
    Function to attach the immediate history of history_length frames to each frame in an array of frame paths.
    :param frame_paths: (np.ndarray) Frame paths.
    :param history_length: (int) Number of frames of history to append to each frame.
    :return: (np.ndarray) Frame paths with attached frame history.
    """
    # pad with first frame so that frames 0 to history_length-1 can be evaluated
    frame_paths = np.concatenate([np.repeat(frame_paths[0], history_length-1), frame_paths])
    
    # for each frame path, attach its immediate history of history_length frames
    frame_paths = [ frame_paths ]
    for l in range(1, history_length):
        frame_paths.append( np.roll(frame_paths[0], shift=-l, axis=0) )
    frame_paths_with_history = np.stack(frame_paths, axis=1) # of size num_clips x history_length
    
    if history_length > 1:
        return frames_with_history[:-(history_length-1)] # frames have wrapped around, remove last (history_length - 1) frames
    else:
        return frames_with_history

def attach_frame_history_tensor(frames, history_length):
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

def unpack_task(task_dict, device, context_to_device=True, target_to_device=False, preload_clips=False):
   
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
  
    if preload_clips:
        return context_clips, context_paths, context_labels, target_clips, target_paths, target_labels, object_list
    else:
        return context_paths, context_paths, context_labels, target_paths, target_paths, target_labels, object_list
