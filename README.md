# ORBIT: A Real-World Few-Shot Dataset for Teachable Object Recognition

This repository contains the code for [ORBIT: A Real-World Few-Shot Dataset for Teachable Object Recognition](https://arxiv.org/abs/2104.03841). It contains code to download and prepare the ORBIT benchmark dataset, run the baseline models on the benchmark dataset, and compute all evaluation metrics.

The code was authored by Daniela Massiceti and built using PyTorch 1.5.0 and Python 3.7.7.

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

# Citation
```
@article{massiceti2021orbit,
  title={{ORBIT: A Real-World Few-Shot Dataset for Teachable Object Recognition}},
  author={Massiceti, Daniela and Zintgraf, Luisa and Bronskill, John and Theodorou, Lida and Harris, Matthew Tobias and Cutrell, Edward and Morrison, Cecily and Hofmann, Katja and Stumpf, Simone},
  journal={arXiv preprint arXiv:2104.03841},
  year={2021}}
```

# Download ORBIT Benchmark Dataset

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
3. Download and prepare the ORBIT benchmark dataset

   This can be done by running the following script which downloads the benchmark dataset, re-sizes all video frames to 84x84, and saves it as `orbit_benchmark_84` in `<folder/to/save/dataset>`. To save other frame sizes, edit the `FRAME_SIZE` variable in the script.
   
   ```
   bash scripts/download_benchmark_dataset.sh <folder/to/save/dataset>
   ```

   Alternatively, the train/validation/test ZIPs can be manually downloaded [here](https://city.figshare.com/articles/dataset/_/14294597). Each should be unzipped as a separate folder into `<folder/to/save/dataset>`. Use `scripts/resize_videos.py` to re-size the frames.
   
4. Summarize dataset statistics
   ```
   python3 scripts/summarize_dataset.py --data_path $ORBIT_BENCHMARK_ROOT --with_modes 
   # to aggregate stats across train, validation, and test collectors, add --combine_modes
   ```
5. Generate dataset plots using `scripts/plot_dataset.ipynb` (uses Plotly)

# Baselines

The following scipts run the baseline models on the Clutter Video Evaluation (CLU-VE) mode on the ORBIT benchmark dataset. For the Clean Video Evaluation (CLE-VE) mode, set `--context_video_type clean --target_video_type clean`.

The baselines were run on 2x Dell Nvidia Tesla V100 32GB GPU. To run on one GPU, remove the `use_two_gpus` flag. To reduce memory requirement, reduce the `clip_length`, `train_context_num_clips`, `train_target_num_clips`, or `test_context_num_clips` flags. For other arguments, see `utils/args.py`. For all other implementation details see Section 5 and Appendix F in the paper.

Note, remember to activate the conda environment (`conda activate orbit-dataset`) or virtual environment. Also, if you are using Windows (or WSL) you will need to set `WORKERS=0` in `data/queues.py` as multi-threaded data loading is not supported. You will also need to [enable longer file paths](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation#enable-long-paths-in-windows-10-version-1607-and-later) as some file names in the dataset are longer than the system limit.

## CNAPs
Implementation of model-based few-shot learner [CNAPs](https://arxiv.org/abs/1906.07697) (Requeima*, Gordon*, Bronskill* et al., _NeurIPS 2019_).

```
python3 cnaps-learner.py --data_path <folder/to/save/dataset>/orbit_benchmark_84 --frame_size 84 \
                         --context_video_type clean --target_video_type clutter \
                         --classifier versa --adapt_features \
                         --learn_extractor \
                         --train_object_cap 10 --with_train_shot_caps \
                         --use_two_gpus 
```

## ProtoNets

Implementation of metric-based few-shot learner [ProtoNets](https://arxiv.org/abs/1703.05175) (Snell et al., _NeurIPS 2017_).

```
python3 protonet-learner.py --data_path <folder/to/save/dataset>/orbit_benchmark_84 --frame_size 84 \
                            --context_video_type clean --target_video_type clutter \
                            --classifier proto \
                            --learn_extractor \
                            --train_object_cap 10 --with_train_shot_caps \
                            --use_two_gpus
```

## MAML

Implementation of optimization-based few-shot learner [MAML](https://arxiv.org/abs/1703.03400) (Finn et al., _ICML 2017_).
```
python3 maml-learner.py --data_path <folder/to/save/dataset>/orbit_benchmark_84 --frame_size 84 \
                        --context_video_type clean --target_video_type clutter \
                        --learn_extractor \
                        --train_object_cap 10 --with_train_shot_caps \v
                        --learning_rate 0.00001 --inner_learning_rate 0.001 \
                        --num_grad_steps 15 \
                        --use_two_gpus
```

## FineTuner

Implementation of transfer learning few-shot learner based on [Tian*, Wang* et al., 2020](https://arxiv.org/abs/2003.11539) and [Chen et al., 2020](https://arxiv.org/abs/2003.04390).
```
python3 finetune-learner.py --data_path <folder/to/save/dataset>/orbit_benchmark_84 --frame_size 84 \
                            --context_video_type clean --target_video_type clutter \
                            --learn_extractor \
                            --train_object_cap 10 --with_train_shot_caps \
                            --inner_learning_rate 0.1 \
                            --num_grad_steps 50 \
                            --use_two_gpus
```
Note, the FineTuner baseline first trains a generic classifier using object cluster (rather than raw object) labels. These clusters can be found in `data/orbit_{train,validation,test}_object_clusters_labels.json` and `data/object_clusters_benchmark.txt`. During validation/testing, a new linear classification layer is appended to the generic feature extractor, and the model is fine-tuned on each task using the raw object labels.

# Download unfiltered ORBIT dataset

Some collectors/objects/videos did not meet the minimum requirement to be included in the ORBIT benchmark dataset. The full unfiltered ORBIT dataset of 4733 videos of 588 objects can be downloaded and saved to `<folder/to/save/dataset>/orbit_unfiltered` by running the following script
```
bash scripts/download_unfiltered_dataset.sh <folder/to/save/dataset>
```
Alternatively, the train/validation/test/other ZIPs can be manually downloaded [here](https://city.figshare.com/articles/dataset/_/14294597). Use `scripts/merge_and_split_benchmark_users.py` to merge the other folder (see script for usage details).

To summarize and plot the unfiltered dataset, use `scripts/summarize_dataset.py` and `scripts/plot_dataset.ipynb` similar to above.

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
