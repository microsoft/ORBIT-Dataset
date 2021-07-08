# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
from data.datasets import UserEpisodicORBITDataset, ObjectEpisodicORBITDataset
from data.samplers import TaskSampler

WORKERS=4

class DatasetQueue:
    """
    Class for a queue of tasks sampled from UserEpisodicORIBTDataset/ObjectEpisodicORBITDataset.

    """
    def __init__(self, tasks_per_user, shuffle):
        """
        Creates instance of DatasetQueue.
        :param tasks_per_user: (int) Number of tasks per user to add to the queue.
        :param shuffle: (bool) If True, shuffle tasks, else do not shuffled.
        :return: Nothing.
        """
        self.tasks_per_user = tasks_per_user
        self.shuffle = shuffle
        self.num_users = None
        self.collate_fn = self.squeeze

    def squeeze(self, batch):
        #assumes batch_size = 1
        squeezed_batch = {} 
        for k,v in batch[0].items():
            if isinstance(v, torch.Tensor):
                squeezed_batch[k] = v.squeeze(0)
            elif isinstance(v, list):
                squeezed_batch[k] = [b.squeeze(0) for b in v]

        return squeezed_batch
    
    def get_num_users(self):
        return self.num_users
    
    def get_cluster_classes(self):
        return self.dataset.cluster_classes
    
    def get_tasks(self):
        return torch.utils.data.DataLoader(
                dataset=self.dataset,
                pin_memory=False,
                num_workers=WORKERS,
                sampler=TaskSampler(self.tasks_per_user, self.num_users, self.shuffle),
                collate_fn=self.collate_fn
                ) 
    
class UserEpisodicDatasetQueue(DatasetQueue):
    def __init__(self, root, frame_size, object_cap, way_method, shot_method, shots, video_types, clip_length, num_clips, subsample_factor, tasks_per_user, test_mode, with_cluster_labels, with_caps, shuffle):
        DatasetQueue.__init__(self, tasks_per_user, shuffle)
        self.dataset = UserEpisodicORBITDataset(root, frame_size, object_cap, way_method, shot_method, shots, video_types, clip_length, num_clips, subsample_factor, test_mode, with_cluster_labels, with_caps)
        self.num_users = self.dataset.num_users
    
class ObjectEpisodicDatasetQueue(DatasetQueue):
    def __init__(self, root, frame_size, object_cap, way_method, shot_method, shots, video_types, clip_length, num_clips, subsample_factor, tasks_per_user, test_mode, with_cluster_labels, with_caps, shuffle):
        DatasetQueue.__init__(self, tasks_per_user, shuffle)
        self.dataset = ObjectEpisodicORBITDataset(root, frame_size, object_cap, way_method, shot_method, shots, video_types, clip_length, num_clips, subsample_factor, test_mode, with_cluster_labels, with_caps)
        self.num_users = self.dataset.num_users    
