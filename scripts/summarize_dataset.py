# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import glob
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str, help="Path to ORBIT dataset root (either unfiltered or benchmark)")
    parser.add_argument("--with_modes", action="store_true", help="ORBIT dataset root has train/validation/test folders")
    parser.add_argument("--combine_modes", action="store_true", help="Summarize stats across train/validation/test folders")
    args = parser.parse_args()

    modes = ['train', 'validation', 'test'] if args.with_modes else ['']
    if not args.combine_modes:
        for mode in modes:
            num_videos_by_user, num_frames_by_user, video_types = get_tallies_by_user(os.path.join(args.data_path, mode))

            count_stats_by_type, num_frames_stats_by_type = {}, {}
            for video_type in video_types:
                count_stats_by_type[video_type]  = compute_count_stats(num_videos_by_user, video_type)
                num_frames_stats_by_type[video_type] = compute_num_frames_stats(num_frames_by_user, video_type)

            print_stats_as_table(mode, len(num_videos_by_user), num_frames_stats_by_type, count_stats_by_type)
    else:
        num_videos_by_user, num_frames_by_user = [], []
        for mode in modes:
            nv, nf, video_types = get_tallies_by_user(os.path.join(args.data_path, mode))
            num_videos_by_user.extend(nv)
            num_frames_by_user.extend(nf) 

        count_stats_by_type, num_frames_stats_by_type = {}, {}
        for video_type in video_types:
            count_stats_by_type[video_type]  = compute_count_stats(num_videos_by_user, video_type)
            num_frames_stats_by_type[video_type] = compute_num_frames_stats(num_frames_by_user, video_type)

        print_stats_as_table(mode, len(num_videos_by_user), num_frames_stats_by_type, count_stats_by_type)

def compute_num_frames_stats(num_frames_by_user, video_type):
    
    num_frames_per_video = []
    num_frames_per_video_per_user = [] 
    min_frames_per_object, max_frames_per_object = [], []
    
    for user in num_frames_by_user:
        avg_num_frames_per_video = []
        for obj in user:
            if video_type in obj and obj[video_type]: #if videos exist for the object/video type
                num_frames_per_video.extend( obj[video_type] )
                avg_num_frames_per_video.extend( obj[video_type] )
                min_frames_per_object.append( np.min( obj[video_type] ))
                max_frames_per_object.append( np.max( obj[video_type] ))
        if avg_num_frames_per_video: #if not empty
            num_frames_per_video_per_user.append( np.mean(avg_num_frames_per_video) )

    num_frames_per_video_stats = [
                    np.mean(num_frames_per_video),
                    np.std(num_frames_per_video),
                    np.percentile(num_frames_per_video, 25),
                    np.percentile(num_frames_per_video, 75),
                    np.min(num_frames_per_video),
                    np.max(num_frames_per_video)
                    ]

    num_frames_per_video_per_user_stats = [
                    np.mean(num_frames_per_video_per_user),
                    np.std(num_frames_per_video_per_user),
                    np.percentile(num_frames_per_video_per_user, 25),
                    np.percentile(num_frames_per_video_per_user, 75),
                    np.min(num_frames_per_video_per_user),
                    np.max(num_frames_per_video_per_user)
                    ]


    stats = {"num_frames_per_video_stats" : num_frames_per_video_stats,
             "num_frames_per_video_per_user_stats": num_frames_per_video_per_user_stats,
             "total_frames": np.sum(num_frames_per_video),
             "min_frames_per_object" : np.min(min_frames_per_object),
             "max_frames_per_object" : np.max(max_frames_per_object)
             }
    
    return stats

def compute_count_stats(num_videos_by_user, video_type):

    num_videos_per_object = []
    num_videos_per_object_per_user = []
    
    for user in num_videos_by_user:
        for obj in user:
            if video_type in obj:
                num_videos_per_object.append( obj[video_type] )
        num_videos_per_object_per_user.append( num_videos_per_object[-len(user):] )

    num_users = len(num_videos_by_user)
    num_videos = np.sum(num_videos_per_object)
    num_objects = len(num_videos_per_object)
    num_videos_per_object_stats = [
                    np.mean(num_videos_per_object), 
                    np.std(num_videos_per_object),
                    np.percentile(num_videos_per_object, 25),
                    np.percentile(num_videos_per_object, 75),
                    np.min(num_videos_per_object),
                    np.max(num_videos_per_object)
                    ]

    num_videos_per_user = [ np.sum(user) for user in num_videos_per_object_per_user ] 
    num_videos_per_user_stats = [
                    np.mean(num_videos_per_user),
                    np.std(num_videos_per_user),
                    np.percentile(num_videos_per_user, 25),
                    np.percentile(num_videos_per_user, 75),
                    np.min(num_videos_per_user),
                    np.max(num_videos_per_user)
                    ]

    num_objects_per_user = [ len(user) for user in num_videos_per_object_per_user ]
    num_objects_per_user_stats = [
                    np.mean(num_objects_per_user),
                    np.std(num_objects_per_user),
                    np.percentile(num_objects_per_user, 25),
                    np.percentile(num_objects_per_user, 75),
                    np.min(num_objects_per_user),
                    np.max(num_objects_per_user)
                    ]

    avg_videos_per_object_per_user = [ np.mean(user) for user in num_videos_per_object_per_user ]
    num_videos_per_object_per_user_stats = [ 
                    np.mean(avg_videos_per_object_per_user),
                    np.std(avg_videos_per_object_per_user),
                    np.percentile(avg_videos_per_object_per_user, 25),
                    np.percentile(avg_videos_per_object_per_user, 75),
                    np.min(avg_videos_per_object_per_user),
                    np.max(avg_videos_per_object_per_user)
                    ]

    stats = {"num_videos" : num_videos, 
             "num_objects": num_objects,
             "num_videos_per_object_stats" : num_videos_per_object_stats,
             "num_videos_per_user_stats" : num_videos_per_user_stats,
             "num_objects_per_user_stats" : num_objects_per_user_stats,
             'num_videos_per_object_per_user_stats' : num_videos_per_object_per_user_stats
             }
    
    return stats

def print_stats_as_table(mode, num_users, num_frames_stats, count_stats):
    """ 
    num_frames_stats is dict with keys as video_type and values as dict of statistics from compute_num_frames_stats()
    counts_stats is dict with keys as video_type and values as dict of statistics from compute_count_stats()
    """

    cr = "\\\ \n"
    tb = " & "
    i = "{:d}"
    f = "{:.1f}"
    FPS=30
    video_types = sorted(num_frames_stats.keys())

    table = tb + "\#objects" + tb + "\#videos" + tb + "\multicolumn{3}{c}{\#videos per object}" + tb + "\multicolumn{3}{c}{\#seconds}" + tb + "\multicolumn{3}{c}{\#frames}" + cr
    table += tb + tb + tb + "mean/std" + tb + "25/75p" + tb + "min/max" + tb + "mean/std" + tb + "25/75p" + tb + "min/max" + cr
    for video_type in video_types:
        c_stats = count_stats[video_type]
        f_stats = num_frames_stats[video_type]
        table += video_type + tb + \
                i.format(c_stats["num_objects"]) + tb + \
                i.format(c_stats["num_videos"]) + tb + \
                f.format(c_stats["num_videos_per_object_stats"][0]) + "/" + f.format(c_stats["num_videos_per_object_stats"][1]) + tb + \
                f.format(c_stats["num_videos_per_object_stats"][2]) + "/" + f.format(c_stats["num_videos_per_object_stats"][3]) + tb + \
                f.format(c_stats["num_videos_per_object_stats"][4]) + "/" + f.format(c_stats["num_videos_per_object_stats"][5]) + tb + \
                f.format(f_stats["num_frames_per_video_stats"][0]) + "/" + f.format(f_stats["num_frames_per_video_stats"][1]) + tb + \
                f.format(f_stats["num_frames_per_video_stats"][2]) + "/" + f.format(f_stats["num_frames_per_video_stats"][3]) + tb + \
                f.format(f_stats["num_frames_per_video_stats"][4]) + "/" + f.format(f_stats["num_frames_per_video_stats"][5]) + cr
    
    for video_type in video_types:
        c_stats = count_stats[video_type]
        f_stats = num_frames_stats[video_type]
        table += video_type + " per user" + tb + \
                f.format(c_stats["num_objects_per_user_stats"][0]) + "/"  + f.format(c_stats["num_objects_per_user_stats"][1]) + tb + \
                f.format(c_stats["num_videos_per_user_stats"][0]) + "/" + f.format(c_stats["num_videos_per_user_stats"][1]) + tb + \
                f.format(c_stats["num_videos_per_object_per_user_stats"][0]) + "/" + f.format(c_stats["num_videos_per_object_per_user_stats"][1]) + tb + \
                f.format(c_stats["num_videos_per_object_per_user_stats"][2]) + "/" + f.format(c_stats["num_videos_per_object_per_user_stats"][3]) + tb + \
                f.format(c_stats["num_videos_per_object_per_user_stats"][4]) + "/" + f.format(c_stats["num_videos_per_object_per_user_stats"][5]) + tb + \
                f.format(f_stats["num_frames_per_video_per_user_stats"][0]) + "/" + f.format(f_stats["num_frames_per_video_per_user_stats"][1]) + tb + \
                f.format(f_stats["num_frames_per_video_per_user_stats"][2]) + "/" + f.format(f_stats["num_frames_per_video_per_user_stats"][3]) + tb + \
                f.format(f_stats["num_frames_per_video_per_user_stats"][4]) + "/" + f.format(f_stats["num_frames_per_video_per_user_stats"][5]) + cr                
    
    print("--"*10)
    total_frames_by_type = "".join( "- # {:} frames: {:}".format(video_type, num_frames_stats[video_type]["total_frames"]) for video_type in num_frames_stats.keys())
    print("{:s} stats - {:d} users {:}".format(mode, num_users, total_frames_by_type))
    print("--"*10)
    print(table)

def get_tallies_by_user(path):
   
    counts_by_user = []
    num_frames_by_user = []
    video_types = []
    for user in os.listdir(path):
        user_dir = os.path.join(path, user)
        if os.path.isdir(user_dir):
            counts_by_obj, num_frames_by_obj = [], []
            for obj in os.listdir(user_dir):
                obj_dir = os.path.join(user_dir, obj)
                counts_by_type, num_frames_by_type = {}, {}
                for video_type in os.listdir(obj_dir):
                    video_types.append(video_type)
                    type_dir = os.path.join(obj_dir, video_type)
                    num_type = len(os.listdir(type_dir))

                    num_frames = []
                    for vid in os.listdir(type_dir):
                        num_frames.append(len(glob.glob(os.path.join(type_dir, vid, "*.jpg"))))
                
                    counts_by_type.update( { video_type : num_type } )
                    num_frames_by_type.update( { video_type : num_frames } )
                    if 'all' in counts_by_type:
                        counts_by_type['all'] += num_type
                        num_frames_by_type['all'] += num_frames.copy()
                    else:
                        counts_by_type.update( { 'all' : num_type })
                        num_frames_by_type.update( { 'all' : num_frames.copy() })
            
                counts_by_obj.append( counts_by_type )
                num_frames_by_obj.append( num_frames_by_type )
        
            counts_by_user.append( counts_by_obj )
            num_frames_by_user.append( num_frames_by_obj )
    
    video_types = list(set(video_types))
    video_types.append('all')
    return counts_by_user, num_frames_by_user, video_types

if __name__ == "__main__":
    main()
