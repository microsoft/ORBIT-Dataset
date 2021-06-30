# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

class ListBatcher():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.list_size = 0

    def _get_number_of_batches(self):
        assert self.list_size > 0, "self.list_size is 0"
        assert self.batch_size > 0, "self.batch_size is 0"
        
        return int(np.ceil(float(self.list_size) / float(self.batch_size)))

    def _set_list_size(self, list_size):
        self.list_size = list_size

    def _get_batch_indices(self, index):
        batch_start_index = index * self.batch_size
        batch_end_index = batch_start_index + self.batch_size
        if batch_end_index > self.list_size:
            batch_end_index = self.list_size
        return range(batch_start_index, batch_end_index)

    def reset(self):
        self.batch_size = None
        self.list_size = None

