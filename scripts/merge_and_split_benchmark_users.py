# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import glob
import json
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str, help="Path to unfiltered ORBIT dataset root")
    parser.add_argument("--split_json", required=True, type=str, default="./data/orbit_benchmark_users_to_split.json", help="Path to orbit_benchmark_users_to_split.json.")
    args = parser.parse_args()

    # merge benchmark users from "other" folder
    merge_users(args) 

    # split users who were merged for the benchmark
    split_users(args)

def merge_users(args):

    user_paths = glob.glob(os.path.join(args.data_path, "P*"))
    users = [os.path.basename(u) for u in user_paths]
    other_user_paths = glob.glob(os.path.join(args.data_path, "other", "P*"))
    other_users = [os.path.basename(u) for u in other_user_paths]

    for (other_user, other_user_path) in zip(other_users, other_user_paths):
        print("Merging {:}".format(other_user))
        if other_user in users:
            user_path = user_paths[ users.index(other_user)]
            user_objects = os.listdir(user_path)
            other_user_objects = os.listdir(other_user_path)
            for obj in os.listdir(other_user_path):
                obj_dir = os.path.join(other_user_path, obj)
                new_obj_dir = os.path.join(user_path, obj)
                for video_type in os.listdir(obj_dir):
                    type_dir = os.path.join(obj_dir, video_type)
                    new_type_dir = os.path.join(new_obj_dir, video_type)
                    for video in os.listdir(type_dir):
                        video_dir = os.path.join(type_dir, video)
                        new_video_dir = os.path.join(new_type_dir, video)
                        shutil.move(video_dir, new_video_dir)
        else:
            new_user_path = os.path.join(args.data_path, other_user)
            shutil.move(other_user_path, new_user_path)

    shutil.rmtree(os.path.join(args.data_path, "other"))

    print("Merged users re-saved to {:}".format(args.data_path))

def split_users(args):

    with open(args.split_json, 'r') as in_file:
        users_to_split = json.load(in_file)

    for benchmark_user, new_users in users_to_split.items():
        print("Splitting {:} into {:}".format(benchmark_user, " ".join(new_users.keys())))
        benchmark_user_dir = os.path.join(args.data_path, benchmark_user)
        for user, user_objs in new_users.items():
            user_dir = os.path.join(args.data_path, user)
            assert not os.path.exists(user_dir), "{:} exists!".format(user_dir)
            os.makedirs(user_dir)
            for obj in user_objs:
                benchmark_obj_dir = os.path.join(benchmark_user_dir, obj)
                assert os.path.exists(benchmark_obj_dir), "{:} does not exists".format(benchmark_obj_dir)
                obj_dir = os.path.join(user_dir, obj)
                shutil.copytree(benchmark_obj_dir, obj_dir)
                for video_type in os.listdir(obj_dir):
                    type_dir = os.path.join(obj_dir, video_type)
                    for video in os.listdir(type_dir):
                        video_dir = os.path.join(type_dir, video)
                        new_video_dir = video_dir.replace(benchmark_user, user)
                        os.rename(video_dir, new_video_dir)
                        frames = glob.glob(os.path.join(new_video_dir, "*.jpg"))
                        for frame in frames:
                            os.rename(frame, frame.replace(benchmark_user, user))
        shutil.rmtree(benchmark_user_dir)
    
    print("Split users re-saved to {:}".format(args.data_path))

if __name__ == "__main__":
    main()
