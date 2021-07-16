# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import glob
import json
import torch
import random
import torchvision
import numpy as np
from torch.utils.data import Dataset

class ORBITDataset(Dataset):
    """
    Base class for ORBIT dataset.
    """
    def __init__(self, root, frame_size, object_cap, way_method, clip_length, subsample_factor, test_mode, with_cluster_labels, with_caps):
        """
        Creates instance of ORBITDataset.
        :param root: (str) Path to train/validation/test folder in ORBIT dataset root folder.
        :param frame_size: (int) Size of video frames.
        :param object_cap: (int or str) Cap on number of objects per user. If 'max', leave uncapped.
        :param way_method: (str) If 'random', select a random number of objects per user. If 'max', select all objects per user.
        :param clip_length: (int) Number of contiguous frames per video clip.
        :param subsample_factor: (int) Factor to subsample video frames before sampling clips.
        :param test_mode: (bool) If True, returns validation/test tasks per user, otherwise returns train tasks per user.
        :param with_cluster_labels: (bool) If True, use object cluster labels, otherwise use raw object labels.
        :param with_caps: (bool) If True, impose caps on the number of videos per object, otherwise leave uncapped.
        :return: Nothing.
        """
        self.root = root
        self.frame_size = frame_size
        self.object_cap = object_cap
        self.way_method = way_method
        self.clip_length = clip_length
        self.subsample_factor = subsample_factor
        self.test_mode = test_mode
        self.with_cluster_labels = with_cluster_labels
        self.with_caps = with_caps
        self.frame_cap = 1000 # limit number of frames in any one video
        self.load_all_users()
    
    def load_all_users(self):
        """
        Function to load data from self.root
        """
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
        """ 
        Function to compute the number of objects to sample for a user.
        :param num_objects: (int) Total number of objects for current user.
        :return: (int) Total number if self.object_cap == 'max' otherwise returns a random number between 2 and total number.
        """
        # all user's objects if object_cap == 'max' else capped by self.object_cap
        max_objects = num_objects if self.object_cap == 'max' else min(num_objects, self.object_cap) 
        min_objects = 2
        if self.way_method == 'random':
            return random.choice(range(min_objects, max_objects + 1))
        elif self.way_method == 'max':
            return max_objects
    
    def sample_videos(self, object_videos):
        """ 
        Function to sample context and target video paths for a given object.
        :param object_videos: (dict::list::str) Dictionary of context and target video paths for an object.
        :return: (list::str, list::str) Sampled context and target video paths for given object according to self.context_type (clean) and self.target_type (clean/clutter).
        """ 
        if self.context_type == self.target_type: # context = clean; target = held-out clean
            num_context_avail = len(object_videos[self.context_type])
            split = min(self.shot_context, num_context_avail-1) # leave at least 1 target video
            context = self.choose_videos(object_videos[self.context_type][:split], self.shot_context, self.shot_method_context, self.context_shot_cap)
            target = self.choose_videos(object_videos[self.context_type][split:], self.shot_target, self.shot_method_target, self.target_shot_cap)
        else: # context = clean; target = clutter
            context = self.choose_videos(object_videos[self.context_type], self.shot_context, self.shot_method_context, self.context_shot_cap)
            target = self.choose_videos(object_videos[self.target_type], self.shot_target, self.shot_method_target, self.target_shot_cap)

        return context, target
 
    def choose_videos(self, videos, required_shots, shot_method, shot_cap):
        """ 
        Function to choose video paths from a list of video paths according to required shots, shot method, and shot cap.
        :param videos: (list::str) List of video paths.
        :param required_shots: (int) Number of videos to select.
        :param shot_method: (str) Method to select videos with options for specific/fixed/random/max - see comments below.
        :param shot_cap: (int) Cap on number of videos to select.
        :return: (list::str) List of selected video paths. 
        """
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
    
    def sample_clips_from_videos(self, video_paths, num_clips):
        """ 
        Function to sample clips from a list of videos.
        :param video_paths: (list::str) List of video paths.
        :param num_clips: (int or str) Number of clips of contiguous frames to sample per video. If 'max', sample all non-overlapping clips.
        :return: (list::np.ndarray, list::int) Frame paths organised in clips of self.clip_length contiguous frames, and video ID for each sampled clip.
        """
        clip_paths, video_ids = [], []
        for video_path in video_paths:
            sampled_clip_paths = self.sample_clips_from_a_video(video_path, num_clips)
            clip_paths.extend(sampled_clip_paths)
            video_ids.extend([self.video2id[video_path] for _ in range(len(sampled_clip_paths))])
        
        return clip_paths, video_ids
    
    def get_video_label(self, obj, label_map):
        """ 
        Function to get mapped object label.
        :param obj: (int) Object ID.
        :param label_map: (dict::int) Dictionary mapping objects to labels.
        :return: (int) Mapped object label.
        """ 
        return label_map[ obj ]
    
    def sample_clips_from_a_video(self, video_path, num_clips):
        """ 
        Function to sample num_clips clips from a single video.
        :param video_path: (str) Path to a single video.
        :param num_clips: (int or str) Number of clips of contiguous frames to sample from the video. If 'max', sample all non-overlapping clips.
        :return: (np.ndarray::str) Frame paths organised in clips of self.clip_length contiguous frames.
        """
        # get all frame paths from video
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*.jpg")))

        # subsample frames by subsample_factor
        subsampled_frame_paths = frame_paths[0:self.frame_cap:self.subsample_factor]
        num_subsampled_frames = len(subsampled_frame_paths)

        if num_clips == 'max': # select all clips from video
            max_num_clips = num_subsampled_frames // self.clip_length
            selected_frame_paths = subsampled_frame_paths[:max_num_clips*self.clip_length]
            sampled_clip_paths = np.array(selected_frame_paths, dtype=str).reshape((max_num_clips, self.clip_length))
        else: # select fixed num_clips from video 
            # if num_required_frames cannot be filled, duplicate random frames
            num_clips = int(num_clips) 
            num_required_frames = self.clip_length*num_clips
            diff = num_required_frames - num_subsampled_frames
            if diff > 0:
                if diff > len(frame_paths):
                    duplicate_frame_paths = random.choices(frame_paths, k=diff) #with replacement
                else:
                    duplicate_frame_paths = random.sample(frame_paths, diff) #without replacement

                subsampled_frame_paths.extend(duplicate_frame_paths)
                subsampled_frame_paths.sort()

            # randomly select num_clips, each of self.clip_length contiguous frames
            frame_idxs = range(self.clip_length, len(subsampled_frame_paths)+1, self.clip_length)
            key_frame_idxs = random.sample(frame_idxs, num_clips)
            sampled_clip_paths = []
            for idx in key_frame_idxs:
                sampled_clip_paths.append(subsampled_frame_paths[idx-self.clip_length:idx])
            sampled_clip_paths = np.array(sampled_clip_paths, dtype=str)
        
        return sampled_clip_paths

    def prepare_task(self, clip_paths, clip_labels, video_ids, test_mode=False):
        """ 
        Function to prepare context/target set for a task.
        :param clip_paths: (list::np.ndarray::str) List of frame paths organised in clips of self.clip_length contiguous frames.
        :param clip_labels: (list::int) List of object labels for each clip.
        :param video_ids: (list::int) List of videos IDs corresponding to clip_paths.
        :param test_mode: (bool) If False, do not shuffle task, otherwise shuffle.
        :return: (np.ndarray::str, torch.Tensor) Frame paths organised in clips and their corresponding video-level labels.
        """
        clip_paths = np.array(clip_paths)
        clip_labels = torch.tensor(clip_labels)
        
        if test_mode:
            clip_paths_by_video, clip_labels_by_video = [], []
            unique_video_ids = np.unique(video_ids)
            for video_id in unique_video_ids:
                idxs = video_ids == video_id
                clip_paths_for_a_video = clip_paths[idxs]
                clip_labels_for_a_video = clip_labels[idxs]
                clip_paths_by_video.append(clip_paths_for_a_video)
                clip_labels_by_video.append(clip_labels_for_a_video)
            return clip_paths_by_video, clip_labels_by_video
        else:
            return self.shuffle_task(clip_paths, clip_labels)
    
    def shuffle_task(self, clip_paths, clip_labels):
        """
        Function to shuffle clips and their object labels.
        :param clip_paths: (np.ndarray::str) Frame paths organised in clips of self.clip_length contiguous frames.
        :param clip_labels: (torch.Tensor) Object labels for each clip.
        :return: (np.ndarray::str, np.ndarray::int) Shuffled clips and their corresponding object labels.
        """
        idxs = np.arange(len(clip_paths))
        random.shuffle(idxs)
        return clip_paths[idxs], clip_labels[idxs]
     
    def get_label_map(self, objects, with_cluster_labels=False):
        """
        Function to get object-to-label map according to if with_cluster_labels is True.
        :param objects: (list::int) List of objects for current user.
        :param with_cluster_labels: (bool) If True, use object cluster labels, otherwise use raw object labels.
        :return: (dict::int) Dictionary mapping objects to labels.
        """
        if with_cluster_labels:
            return self.obj2cluster
        else:
            map_dict = {}
            new_labels = range(len(objects))
            for i, old_label in enumerate(sorted(objects)):
                map_dict[old_label] = new_labels[i]
            return map_dict
    
class UserEpisodicORBITDataset(ORBITDataset):
    """
    Class for user-centric episodic sampling of ORBIT dataset.
    """
    def __init__(self, root, frame_size, object_cap, way_method, shot_methods, shots, video_types, clip_length, num_clips, subsample_factor, test_mode, with_cluster_labels, with_caps):
        """
        Creates instance of UserEpisodicORBITDataset.
        :param root: (str) Path to train/validation/test folder in ORBIT dataset root folder.
        :param frame_size: (int) Size of video frames.
        :param object_cap: (int or str) Cap on number of objects per user. If 'max', leave uncapped.
        :param way_method: (str) If 'random', select a random number of objects per user. If 'max', select all objects per user.
        :param shot_methods: (str, str) Method for sampling videos for context and target sets.
        :param shots: (int, int) Number of videos to sample for context and target sets.
        :param video_types: (str, str) Video types to sample for context and target sets.
        :param clip_length: (int) Number of contiguous frames per video clip.
        :param num_clips: (int or str, int or str) Number of clips to sample per video for context and target sets. If 'max', sample all frames.
        :param subsample_factor: (int) Factor to subsample video frames before sampling clips.
        :param test_mode: (bool) If True, returns validation/test tasks per user, otherwise returns train tasks per user.
        :param with_cluster_labels: (bool) If True, use object cluster labels, otherwise use raw object labels.
        :param with_caps: (bool) If True, impose caps on the number of videos per object, otherwise leave uncapped.
        :return: Nothing.
        """
        ORBITDataset.__init__(self, root, frame_size, object_cap, way_method, clip_length, subsample_factor, test_mode, with_cluster_labels, with_caps)
        
        self.shot_context, self.shot_target = shots
        self.shot_method_context, self.shot_method_target = shot_methods
        self.context_type, self.target_type = video_types
        
        self.context_num_clips, self.target_num_clips = num_clips
        self.context_shot_cap = 15
        self.target_shot_cap = 15
     
    def __getitem__(self, index):
        """
        Function to get a user-centric task as a set of (context and target) clips and labels.
        :param index: (int) Not used.
        :return: Nothing.
        """
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
        
        # for each object, sample context and target clips
        context_clips, target_clips = [], []
        context_labels, target_labels = [], []
        context_clip_video_ids, target_clip_video_ids = [], []
        for i, obj in enumerate(selected_objects):
            context_videos, target_videos = self.sample_videos(self.obj2vids[obj])
            cc, ccvi = self.sample_clips_from_videos(context_videos, self.context_num_clips)
            tc, tcvi = self.sample_clips_from_videos(target_videos, self.target_num_clips)

            context_clips.extend(cc)
            target_clips.extend(tc)

            l = self.get_video_label(obj, label_map)
            context_labels.extend([l for _ in range(len(cc))])
            target_labels.extend([l for _ in range(len(tc))])

            context_clip_video_ids.extend(ccvi)
            target_clip_video_ids.extend(tcvi)
        
        context_clips, context_labels = self.prepare_task(context_clips, context_labels, context_clip_video_ids)
        target_clips, target_labels = self.prepare_task(target_clips, target_labels, target_clip_video_ids, test_mode=self.test_mode)
        
        task_dict = { 
                'context_clips' : context_clips,
                'context_labels' : context_labels,
                'target_clips' : target_clips,
                'target_labels' : target_labels,
        }

        return task_dict
    
class ObjectEpisodicORBITDataset(ORBITDataset):
    """
    Class for object-centric episodic sampling of ORBIT dataset.
    """
    def __init__(self, root, frame_size, object_cap, way_method, shot_methods, shots, video_types, clip_length, num_clips, subsample_factor, test_mode, with_cluster_labels, with_caps):
        """
        Creates instance of ObjectEpisodicORBITDataset.
        :param root: (str) Path to train/validation/test folder in ORBIT dataset root folder.
        :param frame_size: (int) Size of video frames.
        :param object_cap: (int or str) Cap on number of objects per user. If 'max', leave uncapped.
        :param way_method: (str) If 'random', select a random number of objects per user. If 'max', select all objects per user.
        :param shot_methods: (str, str) Method for sampling videos for context and target sets.
        :param shots: (int, int) Number of videos to sample for context and target sets.
        :param video_types: (str, str) Video types to sample for context and target sets.
        :param clip_length: (int) Number of contiguous frames per video clip.
        :param num_clips: (int or str, int or str) Number of clips to sample per video for context and target sets. If 'max', sample all frames.
        :param subsample_factor: (int) Factor to subsample video frames before sampling clips.
        :param test_mode: (bool) If True, returns validation/test tasks per user, otherwise returns train tasks per user.
        :param with_cluster_labels: (bool) If True, use object cluster labels, otherwise use raw object labels.
        :param with_caps: (bool) If True, impose caps on the number of videos per object, otherwise leave uncapped.
        :return: Nothing.
        """
        ORBITDataset.__init__(self, root, frame_size, object_cap, way_method, clip_length, subsample_factor, test_mode, with_cluster_labels, with_caps)
        
        self.shot_context, self.shot_target = shots
        self.shot_method_context, self.shot_method_target = shot_methods
        self.context_type, self.target_type = video_types

        self.context_num_clips, self.target_num_clips = num_clips
        self.context_shot_cap = 15
        self.target_shot_cap = 15
       
    def __getitem__(self, index):
        """
        Function to get a object-centric task as a set of (context and target) clips and labels.
        :param index: (int) Not used.
        :return: Nothing.
        """
        num_objects = len(self.obj2vids)
       
        # select way (number of classes/objects) randomly
        way = self.compute_way(num_objects)
        selected_objects = random.sample(range(0, num_objects), way) # without replacement
        label_map = self.get_label_map(selected_objects, self.with_cluster_labels)
        
        # set caps, for memory purposes (used in training)
        if self.with_caps:
            self.context_shot_cap = 5 if way >=6 else 10
            self.target_shot_cap = 4 if way >=6 else 8

        # for each object, sample context and target clips
        context_clips, target_clips = [], []
        context_labels, target_labels = [], []
        context_clip_video_ids, target_clip_video_ids = [], []
        for i, obj in enumerate(selected_objects):
            context_videos, target_videos = self.sample_videos(self.obj2vids[obj])
            cc, ccvi = self.sample_clips_from_videos(context_videos, self.context_num_clips)
            tc, tcvi = self.sample_clips_from_videos(target_videos, self.target_num_clips)

            context_clips.extend(cc)
            target_clips.extend(tc)

            l = self.get_video_label(obj, label_map)
            context_labels.extend([l for _ in range(len(cc))])
            target_labels.extend([l for _ in range(len(tc))])

            context_clip_video_ids.extend(ccvi)
            target_clip_video_ids.extend(tcvi)
        
        context_clips, context_labels = self.prepare_task(context_clips, context_labels, context_clip_video_ids)
        target_clips, target_labels = self.prepare_task(target_clips, target_labels, target_clip_video_ids, test_mode=self.test_mode)
        
        task_dict = { 
                'context_clips' : context_clips,
                'context_labels' : context_labels,
                'target_clips' : target_clips,
                'target_labels' : target_labels,
        }
        
        return task_dict 
