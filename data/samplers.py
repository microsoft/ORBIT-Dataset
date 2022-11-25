# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import numpy as np
from torch.utils.data.sampler import Sampler

class TaskSampler(Sampler):
    """
    Sampler class for a fixed number of tasks per user/object. 
    """
    def __init__(self, num_tasks_per_item, num_items, shuffle):
        """
        Creates instances of TaskSampler.
        :param num_tasks_per_item: (int) Number of tasks to sample per user/object.
        :param num_items: (int) Total number of users/objects.
        :param shuffle: (bool) If True, shuffle tasks, otherwise do not shuffle.
        :return: Nothing.
        """
        self.num_tasks_per_item = num_tasks_per_item
        self.num_items = num_items
        self.shuffle = shuffle

    def __iter__(self):
        task_ids = []
        for item in range(self.num_items):
            task_ids.extend([item]*self.num_tasks_per_item)
        if self.shuffle:
            random.shuffle(task_ids)
        return iter(task_ids)

    def __len__(self):
        return self.num_items*self.num_tasks_per_item
