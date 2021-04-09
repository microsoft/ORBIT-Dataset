# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import glob
import json
import torch
import random
import torchvision
import numpy as np
import data.transforms as it
from torch.utils.data import Dataset

FRAME_CAP = 1000 # limit number of frames at test time to 1000
FRAME_SIZE = 84 # size of video frames

"""
Base class for ORBIT dataset
"""
class ORBITDataset(Dataset):
    def __init__(self, root, object_cap, way_method, clip_length, subsample_factor, test_mode, with_base_labels):

        self.root = root
        self.object_cap = object_cap
        self.way_method = way_method
        self.clip_length = clip_length
        self.subsample_factor = subsample_factor
        self.test_mode = test_mode
        self.with_base_labels = with_base_labels
        self.transformations = self.get_transformations()
        self.load_all_users()
    
    def load_all_users(self):

        self.users, self.obj2vids, self.obj2name, self.obj2base = [], [], [], []
        self.user2objs = {}
        mode = os.path.basename(self.root)
        with open(os.path.join('data', 'orbit_{:}_object_cluster_labels.json'.format(mode))) as in_file:
            vid2base = json.load(in_file)
        self.base_classes = sorted(set(vid2base.values()))
        obj2base = self.get_label_map(self.base_classes)

        obj_id = 0
        users = [u for u in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, u))]
        for user in sorted(users): # users
            self.users.append(user)
            user_path = os.path.join(self.root, user)
            obj_ids = []
            for obj in sorted(os.listdir(user_path)): #objects per user
                obj_ids.append(obj_id)
                obj_path = os.path.join(user_path, obj)
                videos_by_type = {}
                for video_type in os.listdir(obj_path):
                    videos_by_type[video_type] = []
                    type_path = os.path.join(obj_path, video_type)
                    for vid in sorted(os.listdir(type_path)):
                        obj_base = obj2base [ vid2base[vid] ]
                        vid_path = os.path.join(type_path, vid)
                        videos_by_type[video_type].append(vid_path)
                
                self.obj2vids.append(videos_by_type)
                self.obj2name.append(obj)
                self.obj2base.append(obj_base)
                obj_id += 1
            self.user2objs[user] = obj_ids

        assert (len(self.users) == len(users))
        self.num_users = len(self.users)

    def __len__(self):
        return self.num_users
    
    def get_user_objects(self, user):
        return self.user2objs[ self.users[user] ]
    
    def compute_way(self, num_objects):
        max_objects = min(num_objects, self.object_cap)
        min_objects = 2
        if self.way_method == 'random':
            way = random.choice(range(min_objects, max_objects + 1))
        elif self.way_method == 'max':
            if self.test_mode:
                way = num_objects # all user's objects during meta-test
            else:
                way = max_objects # capped during meta-training
        return way
 
    def sample_videos(self, videos, required_shots, shot_method, shot_cap):

        required_shots = min(required_shots, shot_cap) # first cap for memory purposes
        num_videos = len(videos)
        available_shots = min(required_shots, num_videos) # next cap for video availability purposes

        if shot_method == 'specific': # sample specific videos (1 required_shots = 1st video; 2 required_shots = 1st, 2nd videos, ...)
            return videos[:available_shots]
        elif shot_method == 'fixed': # randomly sample fixed number of videos
            return random.sample(videos, available_shots)
        elif shot_method == 'random': # randomly sample a random number of videos between 1 and num_videos
            max_shots = min(num_videos, shot_cap) # capped for memory reasons
            random_shots = random.choice(range(1, max_shots+1))
            return random.sample(videos, random_shots)
        elif shot_method == 'max': # samples all videos
            max_shots = min(num_videos, shot_cap) # capped for memory reasons
            return random.sample(videos, max_shots)
    
    def get_video_data(self, videos, num_clips):
        frames_list, framepaths_list, num_clips_per_video = [], [], []
        for video_path in videos:
            frame_paths = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
            sampled_frames, sampled_framepaths, vid_num_clips = self.sample_frames(frame_paths, num_clips)
            frames_list.extend(sampled_frames)
            framepaths_list.extend(sampled_framepaths)
            num_clips_per_video.append(vid_num_clips)
        
        return frames_list, framepaths_list, num_clips_per_video
    
    def get_video_label(self, obj, label_map):
        return label_map[ obj ]
    
    def sample_frames(self, frame_paths, num_clips):
       
        sampled_frame_paths, vid_num_clips = self.choose_frames(frame_paths, num_clips)

        # arrange sampled frames in clips of clip_length
        sampled_frames = torch.zeros(vid_num_clips, self.clip_length, 3, FRAME_SIZE, FRAME_SIZE)
        for c in range(vid_num_clips):
            for f in range(self.clip_length):
                sampled_frames[c, f] = self.transformations(sampled_frame_paths[c*self.clip_length+f])

        sampled_frame_paths = np.array(sampled_frame_paths).reshape((vid_num_clips, self.clip_length))
        
        return sampled_frames, sampled_frame_paths, vid_num_clips
    
    def choose_frames(self, frame_paths, num_clips):

        if num_clips == 'max':
            # subsample frames by subsample_factor
            subsampled_frame_paths = frame_paths[0:FRAME_CAP:self.subsample_factor]
            vid_num_clips = len(subsampled_frame_paths) // self.clip_length
            return subsampled_frame_paths[:vid_num_clips*self.clip_length], vid_num_clips
        else: 
            num_required_frames = self.clip_length*num_clips
        
            # first, get a random starting frame
            num_frames = len(frame_paths)
            start_idx = random.choice(range(num_frames-FRAME_CAP)) if num_frames > FRAME_CAP else 0

            # next, subsample frames by subsample_factor
            subsampled_frame_paths = frame_paths[start_idx:FRAME_CAP:self.subsample_factor]
            num_subsampled_frames = len(subsampled_frame_paths)
    
            # if frame_buffer*num_segments cannot be filled, duplicate random frames
            diff = num_required_frames - num_subsampled_frames
            if diff > 0:
                if diff > len(frame_paths):
                    duplicate_frame_paths = random.choices(frame_paths, k=diff) #with replacement
                else:
                    duplicate_frame_paths = random.sample(frame_paths, diff) #without replacement

                subsampled_frame_paths.extend(duplicate_frame_paths)

            # now sample self.num_clips contiguous, non-overlapping clips of self.clip_length each
            subsampled_frame_paths.sort()
            frame_idxs = range(self.clip_length, len(subsampled_frame_paths)+1, self.clip_length)
            key_frame_idxs = random.sample(frame_idxs, num_clips)
            sampled_frame_paths = []
            for idx in key_frame_idxs:
                sampled_frame_paths.extend( subsampled_frame_paths[idx-self.clip_length:idx] )
            return sampled_frame_paths, num_clips

    def prepare_task(self, frames, framepaths, labels, clips_per_video, test_mode=False):
        frames = torch.stack(frames)
        framepaths = np.array(framepaths)
        labels = torch.tensor(labels)
        
        if test_mode:
            idx, flat_clips_per_video, frame_labels = 0, [], []
            for object_videos in clips_per_video: # currently grouped by object
                for num_clips_per_video in object_videos:
                    frame_labels.append( labels[idx].repeat( num_clips_per_video * self.clip_length) )
                    flat_clips_per_video.append( num_clips_per_video )
                    idx += num_clips_per_video
            return frames, list(framepaths.reshape((-1))), frame_labels, flat_clips_per_video
        else:
            return self.shuffle_task(frames, framepaths, labels, clips_per_video)
    
    def flatten_task(self, frames, framepaths, labels, clips_per_video):
        frames = frames.flatten(end_dim=1)
        framepaths = framepaths.reshape((-1))
        labels = labels.view(-1,1).repeat(1, self.clip_length).view(-1)
        return frames, framepaths, labels, clips_per_video
 
    def shuffle_task(self, frames, framepaths, labels, clips_per_video):
        new_order = torch.randperm(len(frames))
        frames = frames[new_order]
        framepaths = framepaths[new_order.numpy()]
        labels = labels[new_order]
        clips_per_video = [] #shuffled tasks don't need clips_per_video markers
        return frames, list(framepaths.reshape((-1))), labels, clips_per_video
    
    def get_transformations(self):
        return it.orbit_benchmark_transform
     
    def get_label_map(self, objects, with_base_labels=False):
        if with_base_labels:
            return self.obj2base
        else:
            map_dict = {}
            new_labels = range(len(objects))
            for i, old_label in enumerate(sorted(objects)):
                map_dict[old_label] = new_labels[i]
            return map_dict
    

"""
ORBIT dataset class for user-centric episodic sampling
"""
class UserEpisodicORBITDataset(ORBITDataset):
    def __init__(self, root, object_cap, way_method, shot_methods, shots, video_types, clip_length, num_clips, subsample_factor, test_mode, with_base_labels):
        ORBITDataset.__init__(self, root, object_cap, way_method, clip_length, subsample_factor, test_mode, with_base_labels)
        
        self.shot_context, self.shot_target = shots
        self.shot_method_context, self.shot_method_target = shot_methods
        self.context_type, self.target_type = video_types
        
        self.context_num_clips, self.target_num_clips = num_clips
        self.context_shot_cap = 15
        self.target_shot_cap = 15
     
    def __getitem__(self, index):

        user = self.users[ index ] # get user (each task == user id)
        user_objects = self.user2objs[user] # get user's objects
        num_user_objects = len(user_objects)

        # select way (number of classes/objects) randomly
        way = self.compute_way(num_user_objects)
        selected_objects = random.sample(user_objects, way) # without replacement
        label_map = self.get_label_map(selected_objects, self.with_base_labels)
        
        # set caps, for memory purposes
        if not self.test_mode:
            self.context_shot_cap = 5 if way >=6 else 10
            self.target_shot_cap = 4 if way >=6 else 8
        
        context_frames, target_frames = [], []
        context_framepaths, target_framepaths = [], []
        context_labels, target_labels = [], []
        context_clips_per_video, target_clips_per_video = [], []
        for i, obj in enumerate(selected_objects):
            context, target = self.sample_shots(self.obj2vids[obj])

            cf, cfp, ccpv = self.get_video_data(context, self.context_num_clips)
            tf, tfp, tcpv = self.get_video_data(target, self.target_num_clips)
            l = self.get_video_label(obj, label_map)

            context_frames.extend(cf)
            target_frames.extend(tf)

            context_framepaths.extend(cfp)
            target_framepaths.extend(tfp)

            context_labels.extend([l for _ in range(len(cf))])
            target_labels.extend([l for _ in range(len(tf))])
            
            context_clips_per_video.append(ccpv)
            target_clips_per_video.append(tcpv)

        context_frames, context_framepaths, context_labels, context_clips_per_video = self.prepare_task(context_frames, context_framepaths, context_labels, context_clips_per_video)
        target_frames, target_framepaths, target_labels, target_clips_per_video = self.prepare_task(target_frames, target_framepaths, target_labels, target_clips_per_video, test_mode=self.test_mode) 
       
        task_dict = { 
                'context_frames' : context_frames,
                'target_frames' : target_frames,
                'context_labels' : context_labels,
                'target_labels' : target_labels,
                'context_framepaths' : context_framepaths,
                'target_framepaths' : target_framepaths,
                'context_clips_per_video' : context_clips_per_video,
                'target_clips_per_video' : target_clips_per_video
        }

        return task_dict
    
    def sample_shots(self, object_videos):
         
        if self.context_type == self.target_type: # context = clean; target = held-out clean
            num_context_avail = len(object_videos[self.context_type])
            split = min(self.shot_context, num_context_avail-1) # leave at least 1 target video
            context = self.sample_videos(object_videos[self.context_type][:split], self.shot_context, self.shot_method_context, self.context_shot_cap)
            target = self.sample_videos(object_videos[self.context_type][split:], self.shot_target, self.shot_method_target, self.target_shot_cap)
        else: # context = clean; target = clutter
            context = self.sample_videos(object_videos[self.context_type], self.shot_context, self.shot_method_context, self.context_shot_cap)
            target = self.sample_videos(object_videos[self.target_type], self.shot_target, self.shot_method_target, self.target_shot_cap)

        return context, target


"""
ORBIT dataset class for object-centric episodic sampling
"""
class ObjectEpisodicORBITDataset(ORBITDataset):
    def __init__(self, root, object_cap, way_method, shot_methods, shots, video_types, clip_length, num_clips, subsample_factor, test_mode, with_base_labels):
        ORBITDataset.__init__(self, root, object_cap, way_method, clip_length, subsample_factor, test_mode, with_base_labels)
        
        self.shot_context, self.shot_target = shots
        self.shot_method_context, self.shot_method_target = shot_methods
        self.context_type, self.target_type = video_types
        self.context_num_clips, self.target_num_clips = num_clips
        
    def __getitem__(self, index):
        
        num_objects = len(self.obj2vids)
       
        # select way (number of classes/objects) randomly
        way = self.compute_way(num_objects)
        selected_objects = random.sample(range(0, num_objects), way) # without replacement
        label_map = self.get_label_map(selected_objects, self.with_base_labels)
        
        # set caps, for memory purposes
        self.context_shot_cap = 5 if way >=6 else 10
        self.target_shot_cap = 4 if way >=6 else 8

        context_frames, target_frames = [], []
        context_framepaths, target_framepaths = [], []
        context_labels, target_labels = [], []
        context_clips_per_video, target_clips_per_video = [], []
        for i, obj in enumerate(selected_objects):
            context, target = self.sample_shots(self.obj2vids[obj])

            cf, cfp, ccpv = self.get_video_data(context, self.context_num_clips)
            tf, tfp, tcpv = self.get_video_data(target, self.target_num_clips)
            l = self.get_video_label(obj, label_map)

            context_frames.extend(cf)
            target_frames.extend(tf)

            context_framepaths.extend(cfp)
            target_framepaths.extend(tfp)

            context_labels.extend([l for _ in range(len(cf))])
            target_labels.extend([l for _ in range(len(tf))])
            
            context_clips_per_video.append(ccpv)
            target_clips_per_video.append(tcpv)

        context_frames, context_framepaths, context_labels, context_clips_per_video = self.prepare_task(context_frames, context_framepaths, context_labels, context_clips_per_video)
        target_frames, target_framepaths, target_labels, target_clips_per_video = self.prepare_task(target_frames, target_framepaths, target_labels, target_clips_per_video, test_mode=self.test_mode) 
       
        task_dict = { 
                'context_frames' : context_frames,
                'target_frames' : target_frames,
                'context_labels' : context_labels,
                'target_labels' : target_labels,
                'context_framepaths' : context_framepaths,
                'target_framepaths' : target_framepaths,
                'context_clips_per_video' : context_clips_per_video,
                'target_clips_per_video' : target_clips_per_video
        }

        return task_dict
    
    def sample_shots(self, object_videos):
         
        if self.context_type == self.target_type: # context = clean; target = held-out clean
            num_context_avail = len(object_videos[self.context_type])
            split = min(self.shot_context, num_context_avail-1) # leave at least 1 target video
            context = self.sample_videos(object_videos[self.context_type][:split], self.shot_context, self.shot_method_context, self.context_shot_cap)
            target = self.sample_videos(object_videos[self.context_type][split:], self.shot_target, self.shot_method_target, self.target_shot_cap)
        else: # context = clean; target = clutter
            context = self.sample_videos(object_videos[self.context_type], self.shot_context, self.shot_method_context, self.context_shot_cap)
            target = self.sample_videos(object_videos[self.target_type], self.shot_target, self.shot_method_target, self.target_shot_cap)

        return context, target
