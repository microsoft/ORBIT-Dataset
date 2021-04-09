# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import numpy as np
from torch.utils.data.sampler import Sampler

class TaskSampler(Sampler):
    def __init__(self, tasks_per_user, num_users, shuffle):
        self.tasks_per_user = tasks_per_user
        self.num_users = num_users
        self.shuffle = shuffle

    def __iter__(self):
        samples = np.repeat(range(self.num_users), self.tasks_per_user)
        if self.shuffle:
            random.shuffle(samples)
        return iter(samples)

    def __len__(self):
        return self.num_users*self.tasks_per_user
