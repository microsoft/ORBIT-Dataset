# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from data.samplers import TaskSampler
from data.datasets import UserEpisodicORBITDataset, ObjectEpisodicORBITDataset

class DatasetQueue:
    """
    Class for a queue of tasks sampled from UserEpisodicORIBTDataset/ObjectEpisodicORBITDataset.

    """
    def __init__(self, num_tasks: int, shuffle: bool, num_workers: int) -> None:
        """
        Creates instance of DatasetQueue.
        :param num_tasks: (int) Number of tasks per user to add to the queue.
        :param shuffle: (bool) If True, shuffle tasks, else do not shuffle.
        :param num_workers: (int) Number of workers to use.
        :return: Nothing.
        """
        self.num_tasks = num_tasks
        self.shuffle = shuffle
        self.num_workers = num_workers

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
                sampler=TaskSampler(self.num_tasks, self.num_users, self.shuffle),
                collate_fn=self.collate_fn
                )

class UserEpisodicDatasetQueue(DatasetQueue):
    def __init__(self, root, way_method, object_cap, shot_method, shots, video_types, subsample_factor, clip_methods, clip_length, frame_size, annotations_to_load, filter_by_annotations, num_tasks, test_mode, with_cluster_labels, with_caps, shuffle, logfile):
        DatasetQueue.__init__(self, num_tasks, shuffle, num_workers=4 if test_mode else 8)
        self.dataset = UserEpisodicORBITDataset(root, way_method, object_cap, shot_method, shots, video_types, subsample_factor, clip_methods, clip_length, frame_size, annotations_to_load, filter_by_annotations, test_mode, with_cluster_labels, with_caps, logfile)
        self.num_users = self.dataset.num_users
    
    def __len__(self):
        return self.dataset.num_users

class ObjectEpisodicDatasetQueue(DatasetQueue):
    def __init__(self, root, way_method, object_cap, shot_method, shots, video_types, subsample_factor, clip_methods, clip_length, frame_size, annotations_to_load, filter_by_annotations, num_tasks, test_mode, with_cluster_labels, with_caps, shuffle, logfile):
        DatasetQueue.__init__(self, num_tasks, shuffle, num_workers=4 if test_model else 8)
        self.dataset = ObjectEpisodicORBITDataset(root, way_method, object_cap, shot_method, shots, video_types, subsample_factor, clip_methods, clip_length, frame_size, annotations_to_load, filter_by_annotations, test_mode, with_cluster_labels, with_caps, logfile)
        self.num_users = self.dataset.num_users
    
    def __len__(self):
        return self.dataset.num_objects
