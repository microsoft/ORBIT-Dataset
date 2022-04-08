# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from typing import Optional
from data.samplers import TaskSampler
from data.datasets import UserEpisodicORBITDataset, ObjectEpisodicORBITDataset

class DatasetQueue:
    """
    Class for a queue of tasks sampled from UserEpisodicORIBTDataset/ObjectEpisodicORBITDataset.

    """
    def __init__(self, tasks_per_user, shuffle, test_mode, override_num_workers: Optional[int]=None):
        """
        Creates instance of DatasetQueue.
        :param tasks_per_user: (int) Number of tasks per user to add to the queue.
        :param shuffle: (bool) If True, shuffle tasks, else do not shuffled.
        :param test_mode: (bool) If True, only return target set for first task per user.
        :param num_workers: (Optional[int]) Number of workers to use. Overrides defaults (4 if test, 8 otherwise).
        :return: Nothing.
        """
        self.tasks_per_user = tasks_per_user
        self.shuffle = shuffle
        self.test_mode = test_mode
        if override_num_workers is None:
            self.num_workers = 4 if self.test_mode else 8
        else:
            self.num_workers = override_num_workers

        self.num_users = None
        self.collate_fn = self.unpack

    def unpack(self, batch):
        #assumes batch_size = 1
        assert len(batch) == 1, "DataLoader needs a batch size of 1!"
        unpacked_batch = {}
        for k,v in batch[0].items():
            unpacked_batch[k] = v
        return unpacked_batch

    def get_num_users(self):
        return self.num_users

    def get_cluster_classes(self):
        return self.dataset.cluster_classes

    def get_tasks(self):
        return torch.utils.data.DataLoader(
                dataset=self.dataset,
                pin_memory=False,
                num_workers=self.num_workers,
                sampler=TaskSampler(self.tasks_per_user, self.num_users, self.shuffle, self.test_mode),
                collate_fn=self.collate_fn
                )

class UserEpisodicDatasetQueue(DatasetQueue):
    def __init__(self, root, way_method, object_cap, shot_method, shots, video_types, subsample_factor, num_clips, clip_length, preload_clips, frame_size, annotations_to_load, tasks_per_user, test_mode, with_cluster_labels, with_caps, shuffle):
        DatasetQueue.__init__(self, tasks_per_user, shuffle, test_mode)
        self.dataset = UserEpisodicORBITDataset(root, way_method, object_cap, shot_method, shots, video_types, subsample_factor, num_clips, clip_length, preload_clips, frame_size, annotations_to_load, test_mode, with_cluster_labels, with_caps)
        self.num_users = self.dataset.num_users

class ObjectEpisodicDatasetQueue(DatasetQueue):
    def __init__(self, root, way_method, object_cap, shot_method, shots, video_types, subsample_factor, num_clips, clip_length, preload_clips, frame_size, annotations_to_load, tasks_per_user, test_mode, with_cluster_labels, with_caps, shuffle):
        DatasetQueue.__init__(self, tasks_per_user, shuffle, test_mode)
        self.dataset = ObjectEpisodicORBITDataset(root, way_method, object_cap, shot_method, shots, video_types, subsample_factor, num_clips, clip_length, preload_clips, frame_size, annotations_to_load, test_mode, with_cluster_labels, with_caps)
        self.num_users = self.dataset.num_users
