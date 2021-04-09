# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

ORBIT UNFILTERED DATASET
4733 videos (3,161,718 frames, 97GB) of 588 objects, collected by 97 people who are blind/low-vision. The dataset is organized in the following file hierachy:
Level 1: Collector. Each folder (e.g. "P999") is a collector. 
Level 2: Object. Each folder is an object from that collector.
Level 3: Video type. Each folder is clean/clutter/clutter-pan for the different video types.
Level 4: Video. Each folder is a video of that video type.
Level 5. Frame. Each file is a frame (as .jpg) from that video.


ORBIT BENCHMARK DATASET
3,822 videos (2,687,934 frames, 83GB) of 386 objects, collected by 67 people who are blind/low-vision. The dataset is organised in the following file hierarchy:
Level 1: Mode. Each folder is train/validation/test.
Level 2: Collector. Each folder (e.g. "P999") is a collector.
Level 3: Object. Each folder is an object from that collector.
Level 4: Video type. Each folder is clean/clutter for the different video types.
Level 5: Video. Each folder is a video of that video type.
Level 6. Frame. Each file is a frame (as .jpg) from that video.

The ORBIT benchmark dataset is extracted from the ORBIT unfiltered dataset.

ORBIT benchmark dataset mean train image [0.50019372 0.43588464 0.39571559] (avg), [0.14545171 0.14291454 0.13843473] (std)
 (computed by running `python scripts/compute_avg_image.py --data_path $ORBIT_BENCHMARK_ROOT`)

