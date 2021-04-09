# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
from data.queues import UserEpisodicDatasetQueue, ObjectEpisodicDatasetQueue

class DataLoader():
    def __init__(self, dataset_info):
        self.train_queue = None
        self.validation_queue = None
        self.test_queue = None
        
        mode = dataset_info['mode']
        if 'train' in mode:
            with_base_labels= True if 'base' in mode else False
            train_config_queue_fn = self.config_user_centric_queue if dataset_info['train_task_type'] == 'user_centric' else self.config_object_centric_queue
            self.train_queue = train_config_queue_fn(
                                        os.path.join(dataset_info['data_path'], 'train'),
                                        dataset_info['object_cap'],
                                        dataset_info['train_way_method'],
                                        dataset_info['train_shot_methods'],
                                        dataset_info['shots'],
                                        dataset_info['video_types'],
                                        dataset_info['clip_length'],
                                        dataset_info['train_num_clips'],
                                        dataset_info['subsample_factor'],
                                        dataset_info['train_tasks_per_user'],
                                        with_base_labels=with_base_labels,
                                        shuffle=True)
            self.validation_queue = self.config_user_centric_queue(
                                        os.path.join(dataset_info['data_path'], 'validation'),
                                        dataset_info['object_cap'],
                                        dataset_info['test_way_method'],
                                        dataset_info['test_shot_methods'],
                                        dataset_info['shots'],
                                        dataset_info['video_types'],
                                        dataset_info['clip_length'],
                                        dataset_info['test_num_clips'],
                                        dataset_info['subsample_factor'],
                                        dataset_info['test_tasks_per_user'],
                                        test_mode=True)
        if 'test' in mode:
            self.test_queue = self.config_user_centric_queue(
                                        os.path.join(dataset_info['data_path'], dataset_info['test_set']),
                                        dataset_info['object_cap'],
                                        dataset_info['test_way_method'],
                                        dataset_info['test_shot_methods'],
                                        dataset_info['shots'],
                                        dataset_info['video_types'],
                                        dataset_info['clip_length'],
                                        dataset_info['test_num_clips'],
                                        dataset_info['subsample_factor'],
                                        dataset_info['test_tasks_per_user'],
                                        test_mode=True)

    def get_train_queue(self):
        return self.train_queue

    def get_validation_queue(self):
        return self.validation_queue
    
    def get_test_queue(self):
        return self.test_queue

    def config_user_centric_queue(self, root, object_cap, way_method, shot_method, shots, video_types, \
                            clip_length, num_clips, subsample_factor, \
                            tasks_per_user, test_mode=False, with_base_labels=False, shuffle=False):
        return UserEpisodicDatasetQueue(root, object_cap, way_method, shot_method, shots, video_types, \
                                clip_length, num_clips, subsample_factor, \
                                tasks_per_user, test_mode, with_base_labels, shuffle)
    
    def config_object_centric_queue(self, root, object_cap, way_method, shot_method, shots, video_types, \
                            clip_length, num_clips, subsample_factor, \
                            tasks_per_user, test_mode=False, with_base_labels=False, shuffle=False):
        return ObjectEpisodicDatasetQueue(root, object_cap, way_method, shot_method, shots, video_types, \
                                clip_length, num_clips, subsample_factor, \
                                tasks_per_user, test_mode, with_base_labels, shuffle)
