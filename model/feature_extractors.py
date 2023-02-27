"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file film.py (https://github.com/cambridge-mlg/dp-fsl/blob/main/src/model.py).

The original license is included below:

MIT License

Copyright (c) 2022 John F. Bronskill

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from timm.models.registry import get_pretrained_cfg
from timm.models.efficientnet import tf_efficientnet_b0, tf_efficientnetv2_s_in21k
from timm.models.vision_transformer import vit_small_patch32_224_in21k, vit_base_patch32_224_in21k, vit_base_patch32_224_clip_laion2b

from model.film import get_film_parameter_names, tag_film_layers

def create_feature_extractor(feature_extractor_name: str, pretrained: bool, with_film: bool=False, learn_extractor: bool=True):
 
    if feature_extractor_name == 'efficientnet_b0':
        pretrained_cfg=get_pretrained_cfg('tf_efficientnet_b0')
        pretrained_cfg['url'] = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_aa-827b6e33.pth'
        feature_extractor = tf_efficientnet_b0(pretrained=pretrained, pretrained_cfg=pretrained_cfg, num_classes=0)
        feature_extractor.output_size = 1280
    elif feature_extractor_name == 'efficientnet_v2_s':
        pretrained_cfg=get_pretrained_cfg('tf_efficientnetv2_s_in21k')
        pretrained_cfg['url'] = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_s_21k-6337ad01.pth'
        feature_extractor = tf_efficientnetv2_s_in21k(pretrained=pretrained, pretrained_cfg=pretrained_cfg, num_classes=0)
        feature_extractor.output_size = 1280
    elif feature_extractor_name == 'vit_s_32':
        pretrained_cfg=get_pretrained_cfg('vit_small_patch32_224_in21k')
        pretrained_cfg['url'] = 'https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz'
        feature_extractor = vit_small_patch32_224_in21k(pretrained=pretrained, pretrained_cfg=pretrained_cfg, num_classes=0)
        feature_extractor.output_size = 384
    elif feature_extractor_name == 'vit_b_32':
        pretrained_cfg=get_pretrained_cfg('vit_base_patch32_224_in21k')
        pretrained_cfg['url'] = 'https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz'
        feature_extractor = vit_base_patch32_224_in21k(pretrained=pretrained, pretrained_cfg=pretrained_cfg, num_classes=0)
        feature_extractor.output_size = 768
    elif feature_extractor_name == 'vit_b_32_clip':
        pretrained_cfg=get_pretrained_cfg('vit_base_patch32_224_clip_laion2b')
        pretrained_cfg['hf_hub_id'] ='laion/CLIP-ViT-B-32-laion2B-s34B-b79K'
        pretrained_cfg['hf_hub_filename'] = 'open_clip_pytorch_model.bin'
        feature_extractor = vit_base_patch32_224_clip_laion2b(pretrained=pretrained, pretrained_cfg=pretrained_cfg, num_classes=0)
        feature_extractor.output_size = 768
    else:
        raise ValueError(f"Invalid feature_extractor_name: {feature_extractor_name}")

    if not learn_extractor:
        freeze_extractor(feature_extractor)
    
    film_param_names = None
    if with_film:
        tag_film_layers(feature_extractor_name, feature_extractor)
        film_param_names = get_film_parameter_names(
            feature_extractor_name=feature_extractor_name,
            feature_extractor=feature_extractor
        )

    return feature_extractor, film_param_names
    
def freeze_extractor(feature_extractor):
    """
    Function that freezes all parameters in the feature extractor.
    :return: Nothing.
    """
    for param in feature_extractor.parameters():
        param.requires_grad = False

