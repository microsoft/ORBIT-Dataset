# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from features.resnet import resnet_18, film_resnet_18

extractors = {
        'resnet_18': resnet_18,
        'film_resnet_18' : film_resnet_18,
        }
