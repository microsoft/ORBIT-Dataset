# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import PIL
import torchvision.transforms as transforms

imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

frame_transform = transforms.Compose([
    lambda x : PIL.Image.open(x),
    transforms.ToTensor(),
    imagenet_normalize
])

