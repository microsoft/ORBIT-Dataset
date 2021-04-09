# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import glob
import time
import argparse
from PIL import Image
from multiprocessing.pool import ThreadPool as Pool

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str, help="Path to ORBIT benchmark dataset root saved by modes")
    parser.add_argument("--save_path", required=True, type=str, help="Path to save resized dataset")
    parser.add_argument("--size", type=int, default=84, help="Target image size.")
    parser.add_argument("--nthreads", type=int, default=12, help="Number of threads.")
    args = parser.parse_args()

    start_time = time.time()
    all_video_dirs = glob.glob( os.path.join(args.data_path, "*/*/*/*/*" ))
    num_videos = len(all_video_dirs)

    pool = Pool(args.nthreads)
    for i, video_dir in enumerate(all_video_dirs):
        pool.apply_async(resize_video_frames, (i, all_video_dirs, args, ))

    pool.close()
    pool.join()
    run_time = (time.time() - start_time) / 60.0
    print('resized videos saved to {:}'.format(os.path.join(args.save_path)))
    print('run time: {:.2f} minutes'.format(run_time))

def resize_video_frames(i, video_dirs, args):

    video_dir = video_dirs[i]
    trunc_frame_dir = video_dir.split('/')[len(args.data_path.split('/')):]
    new_video_dir = os.path.join(args.save_path, '/'.join(trunc_frame_dir))
    os.makedirs(new_video_dir, exist_ok=True)
 
    print("resizing video {:} of {:} - {:}".format(i+1, len(video_dirs), video_dir))
    frames = glob.glob(os.path.join(video_dir, "*"))
    for f in frames:
        pil_image = Image.open(f)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        resized_pil_image = pil_image.resize((args.size, args.size), resample=Image.LANCZOS)
        target_image_path = os.path.join(new_video_dir, os.path.basename(f))
        resized_pil_image.save(target_image_path, quality=95)

if __name__ == "__main__":
    main()
