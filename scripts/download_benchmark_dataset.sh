# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#!/bin/bash

FRAME_SIZE=$2

if [ $FRAME_SIZE = 224 ]
then
   FIGSHARE_TRAIN_URL=https://city.figshare.com/ndownloader/files/28368339
   FIGSHARE_VALIDATION_URL=https://city.figshare.com/ndownloader/files/28368351
   FIGSHARE_TEST_URL=https://city.figshare.com/ndownloader/files/28368072
   export ORBIT_BENCHMARK_ROOT="$1/orbit_benchmark_224"
else
    FIGSHARE_TRAIN_URL=https://city.figshare.com/ndownloader/files/27189155
    FIGSHARE_VALIDATION_URL=https://city.figshare.com/ndownloader/files/27188237
    FIGSHARE_TEST_URL=https://city.figshare.com/ndownloader/files/27346766
   export ORBIT_BENCHMARK_ROOT="$1/orbit_benchmark"
fi

mkdir -p $ORBIT_BENCHMARK_ROOT

# download .zips from FigShare
echo "downloading train.zip..."
wget -O $ORBIT_BENCHMARK_ROOT/train.zip $FIGSHARE_TRAIN_URL
echo "downloading validation.zip..."
wget -O $ORBIT_BENCHMARK_ROOT/validation.zip $FIGSHARE_VALIDATION_URL
echo "downloading test.zip..."
wget -O $ORBIT_BENCHMARK_ROOT/test.zip $FIGSHARE_TEST_URL
echo "ZIPs for train, validation and test collectors downloaded to "$ORBIT_BENCHMARK_ROOT"!"

# unzip .zips
echo "unzipping train.zip..."
unzip -q $ORBIT_BENCHMARK_ROOT/train.zip -d $ORBIT_BENCHMARK_ROOT
echo "unzipping validation.zip..." 
unzip -q $ORBIT_BENCHMARK_ROOT/validation.zip -d $ORBIT_BENCHMARK_ROOT
echo "unzipping test.zip..."
unzip -q $ORBIT_BENCHMARK_ROOT/test.zip -d $ORBIT_BENCHMARK_ROOT
echo "train, validation and test collectors unzipped!"

# removed .zips
rm $ORBIT_BENCHMARK_ROOT/train.zip
rm $ORBIT_BENCHMARK_ROOT/validation.zip
rm $ORBIT_BENCHMARK_ROOT/test.zip

if [ $FRAME_SIZE != 224 ]
then
    # resize frames in all videos to FRAME_SIZE x FRAME_SIZE
    echo "resizing video frames to $FRAME_SIZEx$FRAME_SIZE..."
    RESIZED_ORBIT_BENCHMARK_ROOT=$ORBIT_BENCHMARK_ROOT"_"$FRAME_SIZE
    python3 scripts/resize_videos.py --data_path $ORBIT_BENCHMARK_ROOT --save_path $RESIZED_ORBIT_BENCHMARK_ROOT --size $FRAME_SIZE --nthreads 12
fi
