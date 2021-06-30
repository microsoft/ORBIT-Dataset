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

"""
Base class for ORBIT dataset
"""
class ORBITDataset(Dataset):
    def __init__(self, root, frame_size, object_cap, way_method, clip_length, subsample_factor, test_mode, with_cluster_labels, with_caps):

        self.root = root
        self.frame_size = frame_size
        self.object_cap = object_cap
        self.way_method = way_method
        self.clip_length = clip_length
        self.subsample_factor = subsample_factor
        self.test_mode = test_mode
        self.with_cluster_labels = with_cluster_labels
        self.with_caps = with_caps
        self.transformations = self.get_transformations()
        self.load_all_users()
    
    def load_all_users(self):

        self.users, self.obj2vids, self.obj2name, self.obj2cluster = [], [], [], []
        self.user2objs, self.video2id = {}, {}
        mode = os.path.basename(self.root)
        with open(os.path.join('data', 'orbit_{:}_object_cluster_labels.json'.format(mode))) as in_file:
            vid2cluster = json.load(in_file)
        self.cluster_classes = sorted(set(vid2cluster.values()))
        obj2cluster = self.get_label_map(self.cluster_classes)

        obj_id, vid_id = 0, 0
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
                        obj_cluster = obj2cluster [ vid2cluster[vid] ]
                        vid_path = os.path.join(type_path, vid)
                        videos_by_type[video_type].append(vid_path)
                        self.video2id[vid_path] = vid_id
                        vid_id += 1
                
                self.obj2vids.append(videos_by_type)
                self.obj2name.append(obj)
                self.obj2cluster.append(obj_cluster)
                obj_id += 1
            self.user2objs[user] = obj_ids

        assert (len(self.users) == len(users))
        self.num_users = len(self.users)

    def __len__(self):
        return self.num_users
    
    def get_user_objects(self, user):
        return self.user2objs[ self.users[user] ]
    
    def compute_way(self, num_objects):
        # all user's objects if object_cap == 'max' else capped by self.object_cap
        max_objects = num_objects if self.object_cap == 'max' else min(num_objects, self.object_cap) 
        min_objects = 2
        if self.way_method == 'random':
            return random.choice(range(min_objects, max_objects + 1))
        elif self.way_method == 'max':
            return max_objects
 
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
        frames_list, framepaths_list, video_ids = [], [], []
        for video_path in videos:
            frame_paths = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
            sampled_frames, sampled_framepaths, vid_num_clips = self.sample_frames(frame_paths, num_clips)
            frames_list.extend(sampled_frames)
            framepaths_list.extend(sampled_framepaths)
            video_ids.extend([self.video2id[video_path] for _ in range(vid_num_clips)])
        
        return frames_list, framepaths_list, video_ids
    
    def get_video_label(self, obj, label_map):
        return label_map[ obj ]
    
    def sample_frames(self, frame_paths, num_clips):
       
        sampled_frame_paths, vid_num_clips = self.choose_frames(frame_paths, num_clips)

        # arrange sampled frames in clips of clip_length
        sampled_frames = torch.zeros(vid_num_clips, self.clip_length, 3, self.frame_size, self.frame_size)
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


    def prepare_task(self, frames, framepaths, labels, video_ids, test_mode=False):
        frames = torch.stack(frames)
        framepaths = np.array(framepaths)
        labels = torch.tensor(labels)
        video_ids = torch.tensor(video_ids)
        
        if test_mode:
            frames_by_video, labels_by_video = [], []
            unique_video_ids = torch.unique(video_ids)
            for video_id in unique_video_ids:
                idxs = video_ids == video_id
                video_frames = frames[idxs].flatten(end_dim=1)
                video_labels = labels[idxs].view(-1)
                frames_by_video.append(video_frames)
                labels_by_video.append(video_labels)
            return frames_by_video, labels_by_video
        else:
            return self.shuffle_task(frames, labels)
    
    def shuffle_task(self, frames, labels):
        new_order = torch.randperm(len(frames))
        frames = frames[new_order]
        labels = labels[new_order]
        return frames, labels
    
    def get_transformations(self):
        return it.orbit_benchmark_transform
     
    def get_label_map(self, objects, with_cluster_labels=False):
        if with_cluster_labels:
            return self.obj2cluster
        else:
            map_dict = {}
            new_labels = range(len(objects))
            for i, old_label in enumerate(sorted(objects)):
                map_dict[old_label] = new_labels[i]
            return map_dict
    
    def attach_frame_history(self, frames, labels):

        # expand labels
        labels = labels.view(-1,1).repeat(1, self.clip_length).view(-1)

        # pad with first frame so that first frames 0 to self.clip_length-1 can be evaluated
        frames_0 = frames.narrow(0, 0, 1)
        frames = torch.cat((frames_0.repeat(self.clip_length-1, 1, 1, 1), frames), dim=0)
    
        # for each frame, attach its immediate history of self.clip_length frames
        frames = [ frames ]
        for l in range(1, self.clip_length):
            frames.append( frames[0].roll(shifts=-l, dims=0) )
        frames = torch.stack(frames, dim=1)
        
        # since frames have wrapped around, remove last (num_frames - 1) frames
        return frames[:-(self.clip_length-1)], labels

"""
ORBIT dataset class for user-centric episodic sampling
"""
class UserEpisodicORBITDataset(ORBITDataset):
    def __init__(self, root, frame_size, object_cap, way_method, shot_methods, shots, video_types, clip_length, num_clips, subsample_factor, test_mode, with_cluster_labels, with_caps):
        ORBITDataset.__init__(self, root, frame_size, object_cap, way_method, clip_length, subsample_factor, test_mode, with_cluster_labels, with_caps)
        
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
        label_map = self.get_label_map(selected_objects, self.with_cluster_labels)
        
        # set caps, for memory purposes (used in training)
        if self.with_caps:
            self.context_shot_cap = 5 if way >=6 else 10
            self.target_shot_cap = 4 if way >=6 else 8
        
        context_frames, target_frames = [], []
        context_framepaths, target_framepaths = [], []
        context_labels, target_labels = [], []
        context_video_ids, target_video_ids = [], []
        for i, obj in enumerate(selected_objects):
            context, target = self.sample_shots(self.obj2vids[obj])

            cf, cfp, cvi = self.get_video_data(context, self.context_num_clips)
            tf, tfp, tvi = self.get_video_data(target, self.target_num_clips)
            l = self.get_video_label(obj, label_map)

            context_frames.extend(cf)
            target_frames.extend(tf)

            context_framepaths.extend(cfp)
            target_framepaths.extend(tfp)

            context_labels.extend([l for _ in range(len(cf))])
            target_labels.extend([l for _ in range(len(tf))])
            
            context_video_ids.extend(cvi)
            target_video_ids.extend(tvi)

        context_set, context_labels = self.prepare_task(context_frames, context_framepaths, context_labels, context_video_ids)
        target_set, target_labels = self.prepare_task(target_frames, target_framepaths, target_labels, target_video_ids, test_mode=self.test_mode) 
       
        task_dict = { 
                'context_set' : context_set,
                'context_labels' : context_labels,
                'target_set' : target_set,
                'target_labels' : target_labels
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
    def __init__(self, root, frame_size, object_cap, way_method, shot_methods, shots, video_types, clip_length, num_clips, subsample_factor, test_mode, with_cluster_labels, with_caps):
        ORBITDataset.__init__(self, root, frame_size, object_cap, way_method, clip_length, subsample_factor, test_mode, with_cluster_labels, with_caps)
        
        self.shot_context, self.shot_target = shots
        self.shot_method_context, self.shot_method_target = shot_methods
        self.context_type, self.target_type = video_types

        self.context_num_clips, self.target_num_clips = num_clips
        self.context_shot_cap = 15
        self.target_shot_cap = 15
        
    def __getitem__(self, index):
        
        num_objects = len(self.obj2vids)
       
        # select way (number of classes/objects) randomly
        way = self.compute_way(num_objects)
        selected_objects = random.sample(range(0, num_objects), way) # without replacement
        label_map = self.get_label_map(selected_objects, self.with_cluster_labels)
        
        # set caps, for memory purposes (used in training)
        if self.with_caps:
            self.context_shot_cap = 5 if way >=6 else 10
            self.target_shot_cap = 4 if way >=6 else 8

        context_frames, target_frames = [], []
        context_framepaths, target_framepaths = [], []
        context_labels, target_labels = [], []
        context_video_ids, target_video_ids = [], []
        for i, obj in enumerate(selected_objects):
            context, target = self.sample_shots(self.obj2vids[obj])

            cf, cfp, cvi = self.get_video_data(context, self.context_num_clips)
            tf, tfp, tvi = self.get_video_data(target, self.target_num_clips)
            l = self.get_video_label(obj, label_map)

            context_frames.extend(cf)
            target_frames.extend(tf)

            context_framepaths.extend(cfp)
            target_framepaths.extend(tfp)

            context_labels.extend([l for _ in range(len(cf))])
            target_labels.extend([l for _ in range(len(tf))])
            
            context_video_ids.extend(cvi)
            target_video_ids.extend(tvi)

        context_set, context_labels = self.prepare_task(context_frames, context_framepaths, context_labels, context_video_ids)
        target_set, target_labels = self.prepare_task(target_frames, target_framepaths, target_labels, target_video_ids, test_mode=self.test_mode) 
       
        task_dict = { 
                'context_set' : context_set,
                'context_labels' : context_labels,
                'target_set' : target_set,
                'target_labels' : target_labels
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
