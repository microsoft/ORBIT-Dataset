# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class DatasetFromClipPaths(Dataset):
    def __init__(self, clip_paths, with_labels):
        super().__init__()
        self.with_labels = with_labels
        if self.with_labels:
            self.clip_paths, self.clip_labels = clip_paths
        else:
            self.clip_paths = clip_paths
        
        imagenet_normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                imagenet_normalise
                                ])
        self.loader = self.load_frame

    def __getitem__(self, index):
        clip = []
        for frame_path in self.clip_paths[index]:
            frame = self.loader(frame_path)
            clip.append(self.transform(frame))
        if self.with_labels:
            return torch.stack(clip, dim=0), self.clip_labels[index]
        else:
            return torch.stack(clip, dim=0)

    def load_frame(self, frame_path):
        with open(frame_path, 'rb') as f:
            frame = Image.open(f)
            return frame.convert('RGB')

    def __len__(self):
        return len(self.clip_paths)

def get_clip_loader(clip_paths, batch_size, with_labels=False):
    clips_dataset = DatasetFromClipPaths(clip_paths, with_labels=with_labels)

    return DataLoader(clips_dataset,
                      batch_size=batch_size,
                      num_workers=8,
                      pin_memory=True,
                      persistent_workers=True)

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

    return context_clips, context_labels, target_clips, target_labels
