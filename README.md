# ORBIT: A Real-World Few-Shot Dataset for Teachable Object Recognition

This repository contains code for the following two papers:
- [ORBIT: A Real-World Few-Shot Dataset for Teachable Object Recognition](https://arxiv.org/abs/2104.03841). Code is provided to download and prepare the ORBIT benchmark dataset, and train/test 4 few-shot learning models on this dataset (at 84x84 frames). 
- [Memory Efficient Meta-Learning with Large Images](https://arxiv.org/abs/2107.01105). Code is provided for Large Image and Task Episodic (LITE) training, enabling the models to be trained on large (224x224) frames on a single GPU.

The code was authored by Daniela Massiceti and built using PyTorch 1.5+ and Python 3.7.

<table>
  <tr>
    <td><img src="docs/facemask.PNG" alt="clean frame of facemask" width = 140px></td>
    <td><img src="docs/hairbrush.PNG" alt="clean frame of hairbrush" width = 140px></td>
    <td><img src="docs/keys.PNG" alt="clean frame of keys" width = 140px></td>
    <td><img src="docs/watering can.PNG" alt="clean frame of a watering can" width = 140px></td>
   </tr> 
   <tr>
      <td><img src="docs/facemask_clutter.PNG" alt="clutter frame of facemask" width = 140px></td>
      <td><img src="docs/hairbrush_clutter.PNG" alt="clutter frame of hairbrush" width = 140px></td>
      <td><img src="docs/keys_clutter.PNG" alt="clutter frame of keys" width = 140px></td>
      <td><img src="docs/wateringcan_clutter.PNG" alt="clutter frame of watering can" width = 140px></td>
   </tr>
   <caption style="caption-side:bottom"> <i>Frames from clean (top row) and clutter (bottom row) videos from the ORBIT benchmark dataset</i></caption>
</table>

# Installation

1. Clone or download this repository
2. Install dependencies
   ```
   cd ORBIT-Dataset

   # if using Anaconda
   conda env create -f environment.yml
   conda activate orbit-dataset

   # if using pip
   # pip install -r requirements.txt
   ```

# Download ORBIT Benchmark Dataset


The following script downloads the benchmark dataset into a folder called `orbit_benchmark_<FRAME_SIZE>` at the path `folder/to/save/dataset`. Use `FRAME_SIZE=224` to download the dataset already re-sized to 224x224 frames. For other values of `FRAME_SIZE`, the script will dynamically re-size the frames accordingly:
```
bash scripts/download_benchmark_dataset.sh folder/to/save/dataset FRAME_SIZE
```

Alternatively, the 224x224 train/validation/test ZIPs can be manually downloaded [here](https://city.figshare.com/articles/dataset/_/14294597). Each should be unzipped as a separate train/validation/test folder into `folder/to/save/dataset/orbit_benchmark_224`. The full-size (1080x1080) ZIPs can also be manually downloaded and `scripts/resize_videos.py` can be used to re-size the frames if needed.
   
The following script summarizes the dataset statistics:
```
python3 scripts/summarize_dataset.py --data_path path/to/save/dataset/orbit_benchmark_<FRAME_SIZE> --with_modes 
# to aggregate stats across train, validation, and test collectors, add --combine_modes
```
The Jupyter notebook `scripts/plot_dataset.ipynb` can be used to plot bar charts summarizing the dataset (uses Plotly).

# Training & testing models on ORBIT

The following scripts train and test 4 few-shot learning models on the ORBIT benchmark dataset
- For Clean Video Evaluation (CLE-VE) use `--context_video_type clean --target_video_type clean` 
- For Clutter Video Evaluation (CLU-VE) use `--context_video_type clean --target_video_type clutter`

All models are trained/tested with the arguments below and the defaults specified in `utils/args.py`. These and all other implementation details are described in Section 5 and Appendix F of the [dataset paper](https://arxiv.org/abs/2104.03841). 

Note, before training/testing remember to activate the conda environment (`conda activate orbit-dataset`) or virtual environment. Also, if you are using Windows (or WSL) you will need to set `WORKERS=0` in `data/queues.py` as multi-threaded data loading is not supported. You will also need to [enable longer file paths](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation#enable-long-paths-in-windows-10-version-1607-and-later) as some file names in the dataset are longer than the system limit.

## CNAPs
Implementation of model-based few-shot learner [CNAPs](https://arxiv.org/abs/1906.07697) (Requeima*, Gordon*, Bronskill* et al., _NeurIPS 2019_).

**CNAPs baseline** (see [Table 5](https://arxiv.org/pdf/2104.03841.pdf])) is run with 84x84 frames and a ResNet-18 feature extractor. It is trained/tested on 2x V100 32GB GPUs with training caps on the number of objects/videos per task:
```
python3 single-step-learner.py --data_path folder/to/save/dataset/orbit_benchmark_84 --frame_size 84 \
                               --feature_extractor resnet18 --pretrained_extractor_path features/pretrained/resnet18_imagenet_84.pth \
                               --classifier versa --adapt_features \
                               --context_video_type clean --target_video_type clutter \
                               --train_object_cap 10 --with_train_shot_caps \
                               --use_two_gpus 
```

**CNAPs + LITE** (see [Table 1](https://arxiv.org/pdf/2107.01105.pdf)) is run with 224x224 frames and an EfficientNet-B0 feature extractor. It is trained/tested on 1x Titan RTX 24GB GPU with no training caps on the number of objects/videos per task:
```
python3 single-step-learner.py --data_path folder/to/save/dataset/orbit_benchmark_224 --frame_size 224 \
                         --feature_extractor efficientnetb0 --pretrained_extractor_path features/pretrained/efficientnetb0_imagenet_224.pth \
                         --classifier versa --adapt_features \
                         --context_video_type clean --target_video_type clutter \
                         --with_lite --num_lite_samples 8 --batch_size 8 \
```

## Simple CNAPs
Implementation of model-based few-shot learner [Simple CNAPs](https://arxiv.org/abs/1912.03432) (Bateni et al., _CVPR 2020_).

**Simple CNAPs baseline** (see [Table 1](https://arxiv.org/pdf/2107.01105.pdf)) is run with 84x84 frames and a ResNet-18 feature extractor. It is trained/tested on 2x V100 32GB GPUs with training caps on the number of objects/videos per task:
```
python3 single-step-learner.py --data_path folder/to/save/dataset/orbit_benchmark_84 --frame_size 84 \
                               --feature_extractor resnet18 --pretrained_extractor_path features/pretrained/resnet18_imagenet_84.pth \
                               --classifier mahalanobis --adapt_features \
                               --context_video_type clean --target_video_type clutter \
                               --train_object_cap 10 --with_train_shot_caps \
                               --use_two_gpus 
```

**Simple CNAPs + LITE** (see [Table 1](https://arxiv.org/pdf/2107.01105.pdf)) is run with 224x224 frames and an EfficientNet-B0 feature extractor. It is trained/tested on 1x Titan RTX 24GB GPU with no training caps on the number of objects/videos per task:
```
python3 single-step-learner.py --data_path folder/to/save/dataset/orbit_benchmark_224 --frame_size 224 \
                         --feature_extractor efficientnetb0 --pretrained_extractor_path features/pretrained/efficientnetb0_imagenet_224.pth \
                         --classifier mahalanobis --adapt_features \
                         --context_video_type clean --target_video_type clutter \
                         --with_lite --num_lite_samples 8 --batch_size 8 \
```

## ProtoNets

Implementation of metric-based few-shot learner [ProtoNets](https://arxiv.org/abs/1703.05175) (Snell et al., _NeurIPS 2017_).

**ProtoNets baseline** (see [Table 5](https://arxiv.org/pdf/2104.03841.pdf])) is run with 84x84 frames and a ResNet-18 feature extractor. It is trained/tested on 2x V100 32GB GPUs with training caps on the number of objects/videos per task:
```
python3 single-step-learner.py --data_path folder/to/save/dataset/orbit_benchmark_84 --frame_size 84 \
                              --feature_extractor resnet18 --pretrained_extractor_path features/pretrained/resnet18_imagenet_84.pth \
                              --classifier proto --learn_extractor \
                              --context_video_type clean --target_video_type clutter \
                              --train_object_cap 10 --with_train_shot_caps \
                              --use_two_gpus
```

**ProtoNets + LITE** (see [Table 1](https://arxiv.org/pdf/2107.01105.pdf)) is run with 224x224 frames and an EfficientNet-B0 feature extractor. It is trained/tested on 1x Titan RTX 24GB GPU with no training caps on the number of objects/videos per task:
```
python3 single-step-learner.py --data_path folder/to/save/dataset/orbit_benchmark_224 --frame_size 224 \
                               --feature_extractor efficientnetb0 --pretrained_extractor_path features/pretrained/efficientnetb0_imagenet_224.pth \
                               --classifier proto --learn_extractor \
                               --context_video_type clean --target_video_type clutter \
                               --with_lite --num_lite_samples 16 --batch_size 8
```

## MAML

Implementation of optimization-based few-shot learner [MAML](https://arxiv.org/abs/1703.03400) (Finn et al., _ICML 2017_).

**MAML baseline** (see [Table 5](https://arxiv.org/pdf/2104.03841.pdf])) is run with 84x84 frames and a ResNet-18 feature extractor. It is trained/tested on 1x V100 32GB GPU with training caps on the number of objects/videos per task:
```
python3 maml-learner.py --data_path folder/to/save/dataset/orbit_benchmark_84 --frame_size 84 \
                        --feature_extractor resnet18 --pretrained_extractor_path features/pretrained/resnet18_imagenet_84.pth \
                        --classifier linear --learn_extractor \
                        --context_video_type clean --target_video_type clutter \
                        --train_object_cap 10 --with_train_shot_caps \
                        --learning_rate 0.00001 --inner_learning_rate 0.001 --num_grad_steps 15
```
Note, unlike CNAPs/ProtoNets, it may be possible to train/test on a GPU with less memory by reducing `--batch_size` (or by splitting across two smaller GPUs using `--use_two_gpus` and reducing `--batch_size`).

**MAML on large images** (see [Table 1](https://arxiv.org/pdf/2107.01105.pdf)) is run with 224x224 frames and an EfficientNet-B0 feature extractor. It is trained/tested on 1x Titan RTX 24GB GPU with no training caps on the number of objects/videos per task:
```
python3 maml-learner.py --data_path folder/to/save/dataset/orbit_benchmark_224 --frame_size 224 \
                        --feature_extractor efficientnetb0 --pretrained_extractor_path features/pretrained/efficientnetb0_imagenet_224.pth \
                        --classifier linear --learn_extractor \
                        --context_video_type clean --target_video_type clutter \
                        --learning_rate 0.00001 --inner_learning_rate 0.001 --num_grad_steps 15 \
                        --batch_size 16
```
Note, it is possible to test on a GPU with less memory by reducing `--batch_size`. 

## FineTuner

Implementation of transfer learning few-shot learner based on [Tian*, Wang* et al., 2020](https://arxiv.org/abs/2003.11539). Note, the FineTuner baseline first trains a generic classifier using object cluster (rather than raw object) labels. These clusters can be found in `data/orbit_{train,validation,test}_object_clusters_labels.json` and `data/object_clusters_benchmark.txt`. During validation/testing, a new linear classification layer is appended to the generic feature extractor, and the model is fine-tuned on each task using the raw object labels.

**FineTuner baseline** (see [Table 5](https://arxiv.org/pdf/2104.03841.pdf])) is run with 84x84 frames and a ResNet-18 feature extractor. It is trained/tested on 1x V100 32GB GPU with training caps on the number of objects/videos per task:
```
python3 finetune-learner.py --data_path folder/to/save/dataset/orbit_benchmark_84 --frame_size 84 \
                            --feature_extractor resnet18 --pretrained_extractor_path features/pretrained/resnet18_imagenet_84.pth \
                            --classifier linear --learn_extractor \
                            --context_video_type clean --target_video_type clutter \
                            --train_object_cap 10 --with_train_shot_caps \
                            --inner_learning_rate 0.1 --num_grad_steps 50 
```
Note, like MAML, it is possible to train/test on a GPU with less memory by reducing `--batch_size`. 

**FineTuner on large images** (see [Table 1](https://arxiv.org/pdf/2107.01105.pdf)) is run with 224x224 frames and an EfficientNet-B0 feature extractor. The model used in the paper is not trained on ORBIT, but instead freezes a pre-trained ImageNet extractor and finetunes a new classification layer for each ORBIT test task using standard batch processing on 1x Titan RTX 24GB GPU:
```
python3 finetune-learner.py --data_path folder/to/save/dataset/orbit_benchmark_224 --frame_size 224 \
                            --feature_extractor efficientnetb0 --pretrained_extractor_path features/pretrained/efficientnetb0_imagenet_224.pth \
                            --mode test \
                            --classifier linear \
                            --context_video_type clean --target_video_type clutter \
                            --inner_learning_rate 0.1 --num_grad_steps 50 \
                            --batch_size 16
```
Note, like MAML, it is possible to test on a GPU with less memory by reducing `--batch_size`. 

# GPU and CPU memory requirements

In general, the models take 16-24GB of GPU memory for training and testing. In addition, the models take 16-24GB of CPU memory for training, but can take up to 50-60GB for testing (since all frames from a user's videos are loaded).

The GPU memory requirements can be reduced by:
* Training with LITE (only relevant for CNAPs and ProtoNets and typically only needed for 224x224 or larger images). This can be activated with the `--with_lite` flag. Memory can be further saved by lowering `--num_lite_samples`. 
* Using a smaller `batch_size`. This is relevant for CNAPs and ProtoNets (trained with/without LITE) and also MAML and FineTuner. 
* Lowering the `--clip_length`, `--train_context_num_clips`, `--train_target_num_clips`, or `--test_context_num_clips` arguments.


The CPU memory requirements can be reduced by:
* Reducing the number of data loader workers (see `num_workers` in `data/queues.py`).
* Using the `--no_preload_clips` flag. This will delay the reading of frames from disk into CPU (and then GPU) memory until the very last moment (i.e. when a batch is processed). This can signficiantly reduce the CPU memory required, but comes at the cost of slower training/testing.

# Pre-trained checkpoints

The following checkpoints have been trained on the ORBIT benchmark dataset using the arguments as specified above. The models can be run in test-only mode using the same arguments as above except adding `--mode test` and providing the path to the checkpoint as `--model_path path/to/checkpoint.pt`. In principle, the memory required for testing should be significantly less than training so should be possible on 1x 12-16GB GPU (or CPU with ``--gpu -1``). The ``--batch_size`` flag can be used to further reduce memory requirements.

|   Model   | Frame size | Feature extractor |  Trained with LITE | Trained with clean/clean (context/target) videos | Trained with clean/clutter (context/target) videos |
|:---------:|:----------:|:-----------------:|:------------------:|:-------------:|:-------------------:|
|   CNAPs   |     84     |     ResNet-18     |         N          |[`orbit_cleve_cnaps_resnet18_84.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cleve_cnaps_resnet18_84.pth)|[`orbit_cluve_cnaps_resnet18_84.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_cnaps_resnet18_84.pth)|
|           |     224    |  ResNet-18  |         Y          |[`orbit_cleve_cnaps_resnet18_224_lite.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cleve_cnaps_resnet18_224_lite.pth)|[`orbit_cluve_cnaps_224_resnet18_224_lite.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_cnaps_resnet18_224_lite.pth)|
|           |     224    |  EfficientNet-B0  |         Y          |[`orbit_cleve_cnaps_efficientnetb0_224_lite.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cleve_cnaps_efficientnetb0_224_lite.pth)|[`orbit_cluve_cnaps_224_efficientnetb0_224_lite.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_cnaps_efficientnetb0_224_lite.pth)|
|   Simple CNAPs   |     84     |     ResNet-18     |         N          |[`orbit_cleve_simplecnaps_resnet18_84.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cleve_simplecnaps_resnet18_84.pth)|[`orbit_cluve_simplecnaps_resnet18_84.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_simplecnaps_resnet18_84.pth)|
|           |     224    |  ResNet-18  |         Y          |[`orbit_cleve_simplecnaps_resnet18_224_lite.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cleve_simplecnaps_resnet18_224_lite.pth)|[`orbit_cluve_simplecnaps_224_resnet18_224_lite.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_simplecnaps_resnet18_224_lite.pth)|
|           |     224    |  EfficientNet-B0  |         Y          |[`orbit_cleve_simplecnaps_efficientnetb0_224_lite.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cleve_simplecnaps_efficientnetb0_224_lite.pth)|[`orbit_cluve_simplecnaps_224_efficientnetb0_224_lite.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_simplecnaps_efficientnetb0_224_lite.pth)|
| ProtoNets |     84     |     ResNet-18     |         N          |[`orbit_cleve_protonets_resnet18_84.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cleve_protonets_resnet18_84.pth)|[`orbit_cluve_protonets_resnet18_84.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_protonets_resnet18_84.pth)|
|           |     224    |  ResNet-18  |         Y          |[`orbit_cleve_protonets_resnet18_224_lite.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cleve_protonets_resnet18_224_lite.pth)|[`orbit_cluve_protonets_resnet18_224_lite.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_protonets_resnet18_224_lite.pth)|
|           |     224    |  EfficientNet-B0  |         Y          |[`orbit_cleve_protonets_efficientnetb0_224_lite.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cleve_protonets_efficientnetb0_224_lite.pth)|[`orbit_cluve_protonets_efficientnetb0_224_lite.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_protonets_efficientnetb0_224_lite.pth)|
|    MAML   |     84     |     ResNet-18     |         N          |[`orbit_cleve_maml_resnet18_84.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cleve_maml_resnet18_84.pth)|[`orbit_cluve_maml_resnet18_84.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_maml_resnet18_84.pth)|
|           |     224    |  ResNet-18 |         Y          | [`orbit_cleve_maml_resnet18_224.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cleve_maml_resnet18_224.pth)|[`orbit_cluve_maml_resnet18_224.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_maml_resnet18_224.pth)|
|           |     224    |  EfficientNet-B0  |         Y          | [`orbit_cleve_maml_efficientnetb0_224.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cleve_maml_efficientnetb0_224.pth)|[`orbit_cluve_maml_efficientnetb0_224.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_maml_efficientnetb0_224.pth)|
| FineTuner |     84     |     ResNet-18     |         N          |[`orbit_cleve_finetuner_resnet18_84.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cleve_finetuner_resnet18_84.pth)|[`orbit_cluve_finetuner_resnet18_84.pth`](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_finetuner_resnet18_84.pth)|

# Annotations for ORBIT clutter videos

The bounding boxes for all clutter video frames in the ORBIT benchmark dataset are available in [`data/orbit_clutter_bounding_boxes.zip`](https://github.com/microsoft/ORBIT-Dataset/raw/master/data/orbit_clutter_bounding_boxes.zip). Note, there is _one_ bounding box per frame (i.e. the location of the labelled/target object). Other details:
* The annotations are saved in train/validation/test folders, following the benchmark splits.
* In each folder, there is a JSON per video (e.g. `P177--bag--clutter--Zj_1HvmNWejSbmYf_m4YzxHhSUUl-ckBtQ-GSThX_4E.json`).
* Each JSON contains keys that correspond to all frames in that video.
* For each frame, annotations are saved in the format `{"P177--bag--clutter--Zj_1HvmNWejSbmYf_m4YzxHhSUUl-ckBtQ-GSThX_4E-00001.jpg": {"object_bounding_box": {"x": int, "y": int, "w": int, "h": int}, "object_present": true}, ...}` where `(0,0)` is the top left corner. The coordinates are given for the original 1080x1080 frames, thus `x` and `y` range from `[0,1079]`, and `width` and `height` from `[1,1080]`.
* When there is no object in the frame, the annotations are given as `{"object_bounding_box": null, "object_present": false}`.

You can use the `--frame_annotations` argument to load these annotations. The argument can take the values `object_bounding_box`, `object_present`, or both. This will load the specified annotations (for clutter frames only) and return them in a dictionary. At present, the code does not use these annotations for training/testing. To do so, you will need to return them in the `unpack_task` function in `utils/data.py`.

# Download unfiltered ORBIT dataset

Some collectors/objects/videos did not meet the minimum requirement to be included in the ORBIT benchmark dataset. The full unfiltered ORBIT dataset of 4733 videos (frame size: 1080x1080) of 588 objects can be downloaded and saved to `folder/to/save/dataset/orbit_unfiltered` by running the following script
```
bash scripts/download_unfiltered_dataset.sh folder/to/save/dataset
```
Alternatively, the train/validation/test/other ZIPs can be manually downloaded [here](https://city.figshare.com/articles/dataset/_/14294597). Use `scripts/merge_and_split_benchmark_users.py` to merge the other folder (see script for usage details).

To summarize and plot the unfiltered dataset, use `scripts/summarize_dataset.py` and `scripts/plot_dataset.ipynb` similar to above.

# Citations

### For models trained with LITE:
```
@article{bronskill2021lite,
  title={{Memory Efficient Meta-Learning with Large Images}},
  author={Bronskill, John and Massiceti, Daniela and Patacchiola, Massimiliano and Hofmann, Katja and Nowozin, Sebastian and Turner, Richard E.},
  journal={Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS)},
  year={2021}}
```
### For ORBIT dataset and baselines:
```
@inproceedings{massiceti2021orbit,
  title={{ORBIT: A Real-World Few-Shot Dataset for Teachable Object Recognition}},
  author={Massiceti, Daniela and Zintgraf, Luisa and Bronskill, John and Theodorou, Lida and Harris, Matthew Tobias and Cutrell, Edward and Morrison, Cecily and Hofmann, Katja and Stumpf, Simone},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}}
```

# Contact

To ask questions or report issues, please open an issue on the Issues tab.

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
