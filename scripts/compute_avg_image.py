# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import glob
import argparse
import numpy as np
from PIL import Image

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to ORBIT benchmark dataset root")
    args = parser.parse_args()

    train_video_dirs = glob.glob( os.path.join(args.data_path, "train/*/*/*/*" ))

    avgs = []
    for video_dir in train_video_dirs:
        print ('processing ' + video_dir)
        frames = glob.glob(os.path.join(video_dir, "*.jpg"))
        for f in frames:
            pil_image = Image.open(f)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            arr_image = np.array(pil_image,dtype=np.float)
            mean_image = arr_image.reshape((-1, 3)).mean(axis=0)
            avgs.append(mean_image)

    arr_avgs = np.array(avgs)
    avg = np.mean( arr_avgs, axis=0) / 255.
    std = np.std( arr_avgs, axis=0) / 255.
    print('pixel stats for train frames in {:}: {:} (avg), {:} (std)'.format(args.data_path, avg, std))

if __name__ == "__main__":
    main()
