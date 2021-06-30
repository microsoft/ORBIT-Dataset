# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from features.resnet import resnet_18, resnet_18_84
from features.efficientnet import efficientnet_b0

extractors = {
        'resnet_18': resnet_18,
        'resnet_18_84': resnet_18_84,
        'efficientnet_b0' : efficientnet_b0
        }
