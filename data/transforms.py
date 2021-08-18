# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from PIL import Image
import torchvision.transforms as transforms

def pil_open(path):
    return Image.open(path)

imagenet_normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
orbit_transform = transforms.Compose([
                                        pil_open,
                                        transforms.ToTensor(),
                                        imagenet_normalise
                                    ])

