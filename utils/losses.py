# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn.functional as F

def cross_entropy(test_logits, test_labels):
    return F.cross_entropy(test_logits, test_labels)
