# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import PIL
from PIL import ImageOps
import torchvision.transforms as transforms


""" Pre-processing transforms for ORBIT benchmark dataset.
"""

orbit_benchmark_normalize = transforms.Normalize(mean=[0.500, 0.436, 0.396], std=[0.145, 0.143, 0.138]) # 84

orbit_benchmark_transform = transforms.Compose([
    lambda x : PIL.Image.open(x),
    transforms.ToTensor(),
    orbit_benchmark_normalize
])

