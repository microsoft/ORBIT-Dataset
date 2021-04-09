# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#!/bin/bash

FIGSHARE_TRAIN_URL=https://city.figshare.com/ndownloader/files/27189155
FIGSHARE_VALIDATION_URL=https://city.figshare.com/ndownloader/files/27188237
FIGSHARE_TEST_URL=https://city.figshare.com/ndownloader/files/27346766
FIGSHARE_OTHER_URL=https://city.figshare.com/ndownloader/files/27380768

export ORBIT_UNFILTERED_ROOT="$1/orbit_unfiltered"
mkdir -p $ORBIT_UNFILTERED_ROOT

# download .zips from FigShare
echo "downloading train.zip..."
wget -O $ORBIT_UNFILTERED_ROOT/train.zip $FIGSHARE_TRAIN_URL
echo "downloading validation.zip..."
wget -O $ORBIT_UNFILTERED_ROOT/validation.zip $FIGSHARE_VALIDATION_URL
echo "downloading test.zip..."
wget -O $ORBIT_UNFILTERED_ROOT/test.zip $FIGSHARE_TEST_URL
echo "downloading other.zip..."
wget -O $ORBIT_UNFILTERED_ROOT/other.zip $FIGSHARE_OTHER_URL
echo "ZIPs for train, validation and test collectors downloaded to "$ORBIT_UNFILTERED_ROOT"!"

# unzip .zips
echo "unzipping train.zip..."
unzip -q $ORBIT_UNFILTERED_ROOT/train.zip -d $ORBIT_UNFILTERED_ROOT
echo "unzipping validation.zip..." 
unzip -q $ORBIT_UNFILTERED_ROOT/validation.zip -d $ORBIT_UNFILTERED_ROOT
echo "unzipping test.zip..."
unzip -q $ORBIT_UNFILTERED_ROOT/test.zip -d $ORBIT_UNFILTERED_ROOT
echo "unzipping other.zip..."
unzip -q $ORBIT_UNFILTERED_ROOT/other.zip -d $ORBIT_UNFILTERED_ROOT
echo "train, validation, test and other collectors unzipped!"

# removed .zips
rm $ORBIT_UNFILTERED_ROOT/train.zip
rm $ORBIT_UNFILTERED_ROOT/validation.zip
rm $ORBIT_UNFILTERED_ROOT/test.zip
rm $ORBIT_UNFILTERED_ROOT/other.zip

# move users into root directory
mv $ORBIT_UNFILTERED_ROOT/train/* $ORBIT_UNFILTERED_ROOT
mv $ORBIT_UNFILTERED_ROOT/validation/* $ORBIT_UNFILTERED_ROOT
mv $ORBIT_UNFILTERED_ROOT/test/* $ORBIT_UNFILTERED_ROOT

# remove empty folders
rm -rf $ORBIT_UNFILTERED_ROOT/train
rm -rf $ORBIT_UNFILTERED_ROOT/validation
rm -rf $ORBIT_UNFILTERED_ROOT/test

# merge users from "other" folder then split users who were merged for benchmark dataset
python3 scripts/merge_and_split_benchmark_users.py --data_path $ORBIT_UNFILTERED_ROOT --split_json data/orbit_benchmark_users_to_split.json
