# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
from data.datasets import UserEpisodicORBITDataset, ObjectEpisodicORBITDataset
from data.samplers import TaskSampler

WORKERS=8

class DatasetQueue:
    def __init__(self, tasks_per_user, shuffle):
        self.tasks_per_user = tasks_per_user
        self.shuffle = shuffle
        self.num_users = None
    
    def get_num_users(self):
        return self.num_users
    
    def get_base_classes(self):
        return self.dataset.base_classes
    
    def get_tasks(self):
        return torch.utils.data.DataLoader(
                dataset=self.dataset,
                pin_memory=False,
                num_workers=WORKERS,
                sampler=TaskSampler(self.tasks_per_user, self.num_users, self.shuffle)
                ) 
    
class UserEpisodicDatasetQueue(DatasetQueue):
    def __init__(self, root, object_cap, way_method, shot_method, shots, video_types, clip_length, num_clips, subsample_factor, tasks_per_user, test_mode, with_base_labels, shuffle):
        DatasetQueue.__init__(self, tasks_per_user, shuffle)
        self.dataset = UserEpisodicORBITDataset(root, object_cap, way_method, shot_method, shots, video_types, clip_length, num_clips, subsample_factor, test_mode, with_base_labels)
        self.num_users = self.dataset.num_users
    
class ObjectEpisodicDatasetQueue(DatasetQueue):
    def __init__(self, root, object_cap, way_method, shot_method, shots, video_types, clip_length, num_clips, subsample_factor, tasks_per_user, test_mode, with_base_labels, shuffle):
        DatasetQueue.__init__(self, tasks_per_user, shuffle)
        self.dataset = ObjectEpisodicORBITDataset(root, object_cap, way_method, shot_method, shots, video_types, clip_length, num_clips, subsample_factor, test_mode, with_base_labels)
        self.num_users = self.dataset.num_users    
