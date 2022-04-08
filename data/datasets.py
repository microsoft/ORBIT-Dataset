# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Union
from torch.utils.data import Dataset
import torchvision.transforms.functional as tv_F

class ORBITDataset(Dataset):
    """
    Base class for ORBIT dataset.
    """
    def __init__(self, root, way_method, object_cap, shot_methods, shots, video_types, subsample_factor, num_clips, clip_length, preload_clips, frame_size, annotations_to_load, test_mode, with_cluster_labels, with_caps):
        """
        Creates instance of ORBITDataset.
        :param root: (str) Path to train/validation/test folder in ORBIT dataset root folder.
        :param way_method: (str) If 'random', select a random number of objects per user. If 'max', select all objects per user.
        :param object_cap: (int or str) Cap on number of objects per user. If 'max', leave uncapped.
        :param shot_methods: (str, str) Method for sampling videos for context and target sets.
        :param shots: (int, int) Number of videos to sample for context and target sets.
        :param video_types: (str, str) Video types to sample for context and target sets.
        :param subsample_factor: (int) Factor to subsample video frames before sampling clips.
        :param num_clips: (str, str) Number of clips of contiguous frames to sample from the video. If 'max', sample all non-overlapping clips. If random, sample random number of non-overlapping clips.
        :param clip_length: (int) Number of contiguous frames per video clip.
        :param preload_clips: (bool) If True, preload clips from disk and return as tensors, otherwise return clip paths.
        :param frame_size: (int) Size in pixels of preloaded frames.
        :param annotations_to_load: (list::str) Types of frame annotations to load from disk and return per task.
        :param test_mode: (bool) If True, returns task with target set grouped by video, otherwise returns task with target set not grouped by video.
        :param with_cluster_labels: (bool) If True, use object cluster labels, otherwise use raw object labels.
        :param with_caps: (bool) If True, impose caps on the number of videos per object, otherwise leave uncapped.
        :return: Nothing.
        """
        self.root = root
        self.mode = os.path.basename(self.root)
        self.way_method = way_method
        self.shot_method_context, self.shot_method_target = shot_methods
        self.shot_context, self.shot_target = shots
        self.context_type, self.target_type = video_types
        self.subsample_factor = subsample_factor
        self.context_num_clips, self.target_num_clips = num_clips
        self.clip_length = clip_length
        self.preload_clips = preload_clips
        self.frame_size = frame_size
        self.test_mode = test_mode
        self.with_cluster_labels = with_cluster_labels
        self.with_caps = with_caps
        self.annotations_to_load = sorted(annotations_to_load)
        self.with_annotations = True if annotations_to_load else False
        
        if self.with_annotations:
            self.annotation_dims = {'object_bounding_box': 4 }
            self.annotation_root = os.path.join(os.path.dirname(self.root),  "annotations", f"{self.mode}")       # e.g. /data/orbit_benchmark/annotations/{train,validation,test}
            if not os.path.isdir(self.annotation_root):
                raise IOError(f"Annotation directory {self.annotation_root} does not exist.")

        self.object_cap = object_cap
        self.context_shot_cap = 15
        self.target_shot_cap = 15
        self.clip_cap = 10 # limit number of clips sampled from any one video
        self.frame_cap = 1000 # limit number of frames in any one video
        self.original_frame_size = 1080
        self.normalize_stats = {'mean' : [0.485, 0.456, 0.406], 'std' : [0.229, 0.224, 0.225]} # imagenet mean train frame

        # Setup empty collections.
        self.users = []         # List of users (str)
        self.user2objs = {}     # Dictionary of user (str): list of object ids (int)
        self.obj2name = []      # List of object id (int) to object label (str)
        self.obj2vids = []      # List of dictionaries: {"clean": list of video paths (str), "clutter": list of video paths (str)}
        self.video2id = {}      # Dictionary of video path (str): video id (int)
        self.frame2anns = {}    # Dictionary of frame id (str): annotation information (dict of str to tensor)
        self.vid2frames = {}    # Dictionary of video id (str) to list of valid frame paths

        if self.with_cluster_labels:
            self.obj2cluster = []   # List of object id (int) to cluster id (int)

        self.__load_all_users()        
    
    def __load_all_users(self) -> None:
        
        if self.with_cluster_labels: # setup clusters if we're using object clusters as labels.
            # Load cluster labels for this folder.
            cluster_label_path = os.path.join('data', f"orbit_{self.mode}_object_cluster_labels.json")
            with open(cluster_label_path, 'r') as cluster_label_file:
            # This dictionary shows what cluster each video in the given root belongs to.
                vid2cluster = json.load(cluster_label_file)

            # Get a sorted list of class ints for each cluster.
            # Model output will give index into this list.
            cluster_classes = sorted(set(vid2cluster.values()))

            # We want a list where each object id corresponds to a specific cluster label
            cluster_id_map = self.get_label_map(cluster_classes) #TODO potentially might be an issue with filtering
            
        obj_id, vid_id = 0, 0
        video_types = ['clean', 'clutter']
        for user in tqdm(sorted(os.listdir(self.root)), desc=f"Loading {self.mode} users from {self.root}"): # loop over users
            user_path = os.path.join(self.root, user)
            self.users.append(user)
            obj_ids = []
            for obj_name in sorted(os.listdir(user_path)): # loop over objects per user
                obj_path = os.path.join(user_path, obj_name)
                videos_by_type = {}
                for video_type in video_types: # loop over video types per object [clean and clutter]
                    video_type_path = os.path.join(obj_path, video_type)
                    videos_by_type[video_type] = []
                    for video_name in sorted(os.listdir(video_type_path)): # loop over videos per video type
                        video_path = os.path.join(video_type_path, video_name)
                        self.video2id[video_path] = vid_id
                        videos_by_type[video_type].append(video_path)
                        self.vid2frames[video_path] = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
                        vid_id += 1 
                        if self.with_annotations:
                            video_annotations = self.__load_video_annotations(video_name)
                            self.frame2anns.update(video_annotations)
               
                obj_ids.append(obj_id)
                self.obj2vids.append(videos_by_type)
                self.obj2name.append(obj_name)
                obj_id += 1
                if self.with_cluster_labels:
                    video_name = videos_by_type['clean'][-1] # all videos will have same object label, so just pick 1
                    obj_cluster = cluster_id_map [ vid2cluster[video_name] ]
                    self.obj2cluster.append(obj_cluster)

            self.user2objs[user] = obj_ids
        
        self.num_users = len(self.users)
        self.num_objects = len(self.obj2name)
        print(f"Loaded data summary: {self.num_users} users, {self.num_objects} objects, {len(self.video2id)} videos")
    
    def __load_video_annotations(self, video_name: str) -> Dict[str, Dict[str, Union[bool, torch.Tensor]]]:
        annotation_path = os.path.join(self.annotation_root, f"{video_name}.json")
        with open(annotation_path, 'r') as annotation_file:
            video_annotations = json.load(annotation_file)

        if 'object_bounding_box' in self.annotations_to_load:
            video_annotations = self.__preprocess_bounding_boxes(video_annotations)
        
        return video_annotations
    
    def __preprocess_bounding_boxes(self, video_annotations: Dict[str, Dict[str, Union[bool, torch.Tensor]]]) -> Dict[str, Dict[str, Union[bool, torch.Tensor]]]:

        for frame_id, annotation_dict in video_annotations.items():
            if "object_bounding_box" in annotation_dict and annotation_dict["object_bounding_box"] is not None:
                bbox = annotation_dict["object_bounding_box"]
                bbox = torch.tensor([bbox["x"], bbox["y"], bbox["w"], bbox["h"]])
                # Resize to fit current frame size.
                bbox = ((bbox / self.original_frame_size) * self.frame_size).int()
                # Clamp box to fit within frame.
                bbox[0:2] = torch.clamp(bbox[0:2], 0, self.frame_size - 1)
                bbox[2:4] = torch.clamp(bbox[2:4], 1, self.frame_size)
                video_annotations[frame_id]["object_bounding_box"] = bbox

        return video_annotations

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
        if self.context_type == self.target_type == 'clean': # context = clean; target = held-out clean
            num_context_avail = len(object_videos['clean'])
            split = min(5, num_context_avail-1) # minimum of 5 context, unless not enough then leave at least 1 target video and use remaining as context
            context = self.choose_videos(object_videos['clean'][:split], self.shot_context, self.shot_method_context, self.context_shot_cap)
            target = self.choose_videos(object_videos['clean'][split:], self.shot_target, self.shot_method_target, self.target_shot_cap)
        elif self.context_type == 'clean' and self.target_type == 'clutter': # context = clean; target = clutter
            context = self.choose_videos(object_videos['clean'], self.shot_context, self.shot_method_context, self.context_shot_cap)
            target = self.choose_videos(object_videos['clutter'], self.shot_target, self.shot_method_target, self.target_shot_cap)

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

    def sample_clips_from_videos(self, video_paths: List[str], num_clips: str):
        """
        Function to sample clips from a list of videos.
        :param video_paths: (list::str) List of video paths.
        :param num_clips: (str) Number of clips of contiguous frames to sample from the video. If 'max', sample all non-overlapping clips. If random, sample random number of non-overlapping clips.
        :return: (list::torch.Tensor, list::np.ndarray, list::torch.Tensor, list::int) Frame data, paths, and annotations organised in clips of self.clip_length contiguous frames, and video ID for each sampled clip.
        """
        clips, paths, video_ids = [], [], []
        annotations = { ann: [] for ann in self.annotations_to_load }
        for video_path in video_paths:
            sampled_paths = self.sample_clips_from_a_video(video_path, num_clips)
            paths.extend(sampled_paths)

            if self.preload_clips:
                sampled_clips = self.load_clips(sampled_paths)
                clips += sampled_clips
            
            if self.with_annotations:
                sampled_annotations = self.load_annotations(sampled_paths)
                annotations = self.extend_ann_dict(annotations, sampled_annotations)

            video_ids.extend([self.video2id[video_path]] * len(sampled_paths))

        return clips, paths, video_ids, annotations
    
    def extend_ann_dict(self, dest_dict, src_dict):
        """
        Function to extend all lists within annotation dictionary.
        :param dest_dict: (dict::list) Dictionary of lists to extend.
        :param src_dict: (dict::list) Dictionary of lists to add.
        :return: (dict::list) Dictionary of extended lists.
        """
        for ann in dest_dict.keys():
            dest_dict[ann].extend(src_dict[ann])

        return dest_dict

    def load_clips(self, paths):
        """
        Function to load clips from disk into tensors.
        :param paths: (np.ndarray::str) Frame paths organised in clips of self.clip_length contiguous frames.
        :return: (torch.Tensor) Clip data.
        """
        num_clips, clip_length = paths.shape
        assert clip_length == self.clip_length
        loaded_clips = torch.zeros(num_clips, clip_length, 3, self.frame_size, self.frame_size)

        for clip_idx in range(num_clips):
            for frame_idx in range(clip_length):
                frame_path = paths[clip_idx, frame_idx]
                loaded_clips[clip_idx, frame_idx] = self.load_and_transform_frame(frame_path)

        return loaded_clips
    
    def load_annotations(self, paths: np.ndarray) -> torch.Tensor:
        """
        Function to load frame annotations, arrange in clips, from disk.
        :param paths: (np.ndarray::str) Frame paths organised in clips of self.clip_length contiguous frames.
        :return: (torch.Tensor) Frame annotations arranged in clips.
        """
        num_clips, clip_length = paths.shape
        assert clip_length == self.clip_length

        loaded_annotations = {
            annotation : torch.empty(num_clips, clip_length, self.annotation_dims.get(annotation, 1))     # The dimension defaults to 1 unless specified.
            for annotation in self.annotations_to_load }

        for clip_idx in range(num_clips):
            for frame_idx in range(clip_length):
                frame_path = paths[clip_idx, frame_idx]
                frame_name = os.path.basename(frame_path)
                for annotation in self.annotations_to_load:
                    if annotation in self.frame2anns[frame_name] and self.frame2anns[frame_name][annotation] is not None:
                        frame_anns = self.frame2anns[frame_name]
                        loaded_annotations[annotation][clip_idx, frame_idx] = frame_anns[annotation]
                    else:
                        loaded_annotations[annotation][clip_idx, frame_idx] = float('nan')

        return loaded_annotations

    def load_and_transform_frame(self, frame_path):
        """
        Function to load and transform frame.
        :param frame_path: (str) str to frame.
        :return: (torch.Tensor) Loaded and transformed frame.
        """
        frame = Image.open(frame_path)
        frame = tv_F.to_tensor(frame)
        frame = tv_F.normalize(frame, mean=self.normalize_stats['mean'], std=self.normalize_stats['std'])
        return frame

    def sample_clips_from_a_video(self, video_path: str, num_clips: int) -> np.ndarray:
        """
        Function to sample num_clips clips from a single video.
        :param video_path: (str) str to a single video.
        :param num_clips: (str) Number of clips of contiguous frames to sample from the video. If 'max', sample all non-overlapping clips. If random, sample random number of non-overlapping clips.
        :return: (np.ndarray::str) Frame paths organised in clips of self.clip_length contiguous frames.
        """
        # get all frame paths from video
        frame_paths = self.vid2frames[video_path]

        # subsample frames by subsample_factor
        subsampled_frame_paths = frame_paths[0:self.frame_cap:self.subsample_factor]

	# if not divisible by self.clip_length, pad with last frame until it is
        spare_frames = len(subsampled_frame_paths) % self.clip_length
        subsampled_frame_paths.extend([subsampled_frame_paths[-1]] * (self.clip_length - spare_frames))

        num_subsampled_frames = len(subsampled_frame_paths)
        max_num_clips = num_subsampled_frames // self.clip_length
        assert num_subsampled_frames % self.clip_length == 0

        if num_clips == 'max': # select all non_overlapping clips from video
            sampled_paths = subsampled_frame_paths[:max_num_clips*self.clip_length]
            sampled_paths = np.array(sampled_paths).reshape((max_num_clips, self.clip_length))
        elif num_clips == 'random': # select random number of non-overlapping clips from video
            capped_num_clips = min(max_num_clips, self.clip_cap)
            random_num_clips = random.choice(range(1, capped_num_clips+1))

            frame_idxs = range(self.clip_length, len(subsampled_frame_paths)+1, self.clip_length)
            key_frame_idxs = random.sample(frame_idxs, random_num_clips)
            sampled_paths = []
            for idx in key_frame_idxs:
                sampled_paths.append(subsampled_frame_paths[idx-self.clip_length:idx])
            sampled_paths = np.array(sampled_paths)
        else:
            raise ValueError(f"num_clips should be 'max' or 'random', but was {num_clips}")

        return sampled_paths # shape (num_clips, clip_length)
   
    def prepare_set(self, clips, paths, labels, annotations, video_ids, test_mode=False):
        """
        Function to prepare context/target set for a task.
        :param clips: (list::torch.Tensor) List of frame data organised in clips of self.clip_length contiguous frames.
        :param paths: (list::np.ndarray::str) List of frame paths organised in clips of self.clip_length contiguous frames.
        :param labels: (list::int) List of object labels for each clip.
        :param annotations: (dict::list::torch.Tensor) Dictionary of annotations for each clip.
        :param video_ids: (list::int) List of videos IDs corresponding to paths.
        :param test_mode: (bool) If False, do not shuffle task, otherwise shuffle.
        :return: (torch.Tensor or list::torch.Tensor, np.ndarray::str or list::np.ndarray, torch.Tensor or list::torch.Tensor, dict::torch.Tensor or list::dict::torch.Tensor) Frame data, paths, video-level labels and annotations organised in clips (if train) or grouped and flattened by video (if test/validation).
        """
        clips = torch.stack(clips) if self.preload_clips else torch.tensor(clips)
        paths = np.array(paths)
        labels = torch.tensor(labels)
        annotations = { ann: torch.stack(annotations[ann]) for ann in self.annotations_to_load }

        if test_mode: # group by video
            frames_by_video, paths_by_video, labels_by_video, annotations_by_video = [], [], [], []
            unique_video_ids = np.unique(video_ids)
            for video_id in unique_video_ids:
                # get all clips belonging to current video
                idxs = video_ids == video_id
                # flatten frames and paths from current video (assumed to be sorted)
                video_frames = clips[idxs].flatten(end_dim=1) if self.preload_clips else None
                video_paths = paths[idxs].reshape(-1)
                frames_by_video.append(video_frames)
                paths_by_video.append(video_paths)
                # all clips from the same video have the same label, so just return 1
                video_label = labels[idxs][0]
                labels_by_video.append(video_label)
                # get all frame annotations for current video
                video_anns = { ann : annotations[ann][idxs].flatten(end_dim=1) for ann in self.annotations_to_load } if self.with_annotations else None 
                annotations_by_video.append(video_anns)
            return frames_by_video, paths_by_video, labels_by_video, annotations_by_video
        else:
            return self.shuffle_set(clips, paths, labels, annotations)

    def shuffle_set(self, clips, paths, labels, annotations):
        """
        Function to shuffle clips and their object labels.
        :param clips: (torch.Tensor) Frame data organised in clips of self.clip_length contiguous frames.
        :param paths: (np.ndarray::str) Frame paths organised in clips of self.clip_length contiguous frames.
        :param labels: (torch.Tensor) Object labels for each clip.
        :param annotations: (dict::torch.Tensor) Frame annotations organised in clips of self.clip_length contiguous frames.
        :return: (torch.Tensor, np.ndarray::str, torch.Tensor, dict::torch.Tensor) Shuffled clips and their corresponding paths, object labels and annotations.
        """
        idxs = np.arange(len(paths))
        random.shuffle(idxs)
        if self.preload_clips:
            if self.with_annotations:
                return clips[idxs], paths[idxs], labels[idxs], { ann : annotations[ann][idxs] for ann in self.annotations_to_load }
            else:
                return clips[idxs], paths[idxs], labels[idxs], annotations
        else:
            if self.with_annotations:
                return clips, paths[idxs], labels[idxs], { ann : annotations[ann][idxs] for ann in self.annotations_to_load }
            else:
                return clips, paths[idxs], labels[idxs], annotations

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
            for i, old_label in enumerate(objects):
                map_dict[old_label] = new_labels[i]
            return map_dict

    def sample_task(self, task_objects, with_target_set, user_id=''):

        # select way (number of classes/objects) randomly
        num_objects = len(task_objects)
        way = self.compute_way(num_objects)
        selected_objects = sorted(random.sample(task_objects, way)) # without replacement
        label_map = self.get_label_map(selected_objects, self.with_cluster_labels)

        # set caps, for memory purposes (used in training)
        if self.with_caps:
            self.context_shot_cap = 5 if way >=6 else 10
            self.target_shot_cap = 4 if way >=6 else 8

        # for each object, sample context and target sets
        obj_list = []
        context_clips, target_clips = [], []
        context_paths, target_paths = [], []
        context_labels, target_labels = [], []
        context_video_ids, target_video_ids = [], []
        context_annotations = { ann : [] for ann in self.annotations_to_load}
        target_annotations = { ann : [] for ann in self.annotations_to_load}
        for obj in selected_objects:
            label = label_map[obj]
            obj_name = self.obj2name[obj]
            obj_list.append(obj_name)

            context_videos, target_videos = self.sample_videos(self.obj2vids[obj])
            cc, cp, cvi, ca = self.sample_clips_from_videos(context_videos, self.context_num_clips)
            context_clips.extend(cc)
            context_paths.extend(cp)
            context_labels.extend([label for _ in range(len(cp))])
            context_video_ids.extend(cvi)
            context_annotations = self.extend_ann_dict(context_annotations, ca)

            if with_target_set:
                tc, tp, tvi, ta = self.sample_clips_from_videos(target_videos, self.target_num_clips)
                target_clips.extend(tc)
                target_paths.extend(tp)
                target_labels.extend([label for _ in range(len(tp))])
                target_video_ids.extend(tvi)
                target_annotations = self.extend_ann_dict(target_annotations, ta)

        context_clips, context_paths, context_labels, context_annotations = self.prepare_set(context_clips, context_paths, context_labels, context_annotations, context_video_ids)
        if with_target_set:
            target_clips, target_paths, target_labels, target_annotations = self.prepare_set(target_clips, target_paths, target_labels, target_annotations, target_video_ids, test_mode=self.test_mode)

        task_dict = {
            # Data required for train / test
            'context_clips': context_clips,                                     # Tensor of shape (num_context_clips, clip_length, channels, height, width), dtype float32
            'context_paths': context_paths,                                     # Numpy array of shape (num_context_clips, clip_length), dtype str
            'context_labels': context_labels,                                   # Tensor of shape (num_context_clips,), dtype int64
            'context_annotations': context_annotations,                         # Dictionary. Empty if no annotations present. TODO: Add info for when annotations are present.
            'target_clips': target_clips,                                       # If train, tensor of shape (num_target_clips, clip_length, channels, height, width), dtype float32. If test/validation, list of length (num_target_videos_for_user) of tensors, each of shape (num_video_frames, channels, height, width), dtype float32
            'target_paths': target_paths,                                       # If train, numpy array of shape (num_target_clips, clip_length), dtype str. If test/validation, list of length (num_target_videos_for_user) of numpy arrays, each of shape (num_video_frames,)), dtype str
            'target_labels': target_labels,                                     # If train, tensor of shape (num_target_clips,), dtype int64. If test/validation, list of length (num_target_videos_for_user) of tensors, each of shape (1,), dtype int64
            'target_annotations': target_annotations,                           # Dictionary. Empty if no annotations present. TODO: Add info for when annotations are present.
            # Extra information, to be used in logging and results.
            'object_list': obj_list,                                             # Ordered list of strings for all objects in this task.
            'user_id': user_id                                                  # User ID ('' if ObjectEpisodicORBITDataset as task sampled from > 1 user), dtype string
        }
        return task_dict

class UserEpisodicORBITDataset(ORBITDataset):
    """
    Class for user-centric episodic sampling of ORBIT dataset.
    """
    def __init__(self, root, way_method, object_cap, shot_methods, shots, video_types, subsample_factor, num_clips, clip_length, preload_clips, frame_size, annotations_to_load, test_mode, with_cluster_labels, with_caps):
        """
        Creates instance of UserEpisodicORBITDataset.
        :param root: (str) Path to train/validation/test folder in ORBIT dataset root folder.
        :param way_method: (str) If 'random', select a random number of objects per user. If 'max', select all objects per user.
        :param object_cap: (int or str) Cap on number of objects per user. If 'max', leave uncapped.
        :param shot_methods: (str, str) Method for sampling videos for context and target sets.
        :param shots: (int, int) Number of videos to sample for context and target sets.
        :param video_types: (str, str) Video types to sample for context and target sets.
        :param subsample_factor: (int) Factor to subsample video frames before sampling clips.
        :param num_clips: (str, str) Number of clips of contiguous frames to sample from the video. If 'max', sample all non-overlapping clips. If random, sample random number of non-overlapping clips.
        :param clip_length: (int) Number of contiguous frames per video clip.
        :param preload_clips: (bool) If True, preload clips from disk and return as tensors, otherwise return clip paths.
        :param frame_size: (int) Size in pixels of preloaded frames.
        :param annotations_to_load: (list::str) Types of frame annotations to load from disk and return per task.
        :param test_mode: (bool) If True, returns task with target set grouped by video, otherwise returns task with target set not grouped by video.
        :param with_cluster_labels: (bool) If True, use object cluster labels, otherwise use raw object labels.
        :param with_caps: (bool) If True, impose caps on the number of videos per object, otherwise leave uncapped.
        :return: Nothing.
        """
        ORBITDataset.__init__(self, root, way_method, object_cap, shot_methods, shots, video_types, subsample_factor, num_clips, clip_length, preload_clips, frame_size, annotations_to_load, test_mode, with_cluster_labels, with_caps)

    def __getitem__(self, index):
        """
        Function to get a user-centric task as a set of (context and target) clips and labels.
        :param index: (tuple) Task ID and whether to load task target set.
        :return: (dict) Context and target set data for task.
        """

        task_id, with_target_set = index
        user = self.users[task_id] # get user (each task == user id)
        user_objects = self.user2objs[user] # get user's objects
        return self.sample_task(user_objects, with_target_set, user)

class ObjectEpisodicORBITDataset(ORBITDataset):
    """
    Class for object-centric episodic sampling of ORBIT dataset.
    """
    def __init__(self, root, way_method, object_cap, shot_methods, shots, video_types, subsample_factor, num_clips, clip_length, preload_clips, frame_size, annotations_to_load, test_mode, with_cluster_labels, with_caps):
        """
        Creates instance of ObjectEpisodicORBITDataset.
        :param root: (str) Path to train/validation/test folder in ORBIT dataset root folder.
        :param way_method: (str) If 'random', select a random number of objects per user. If 'max', select all objects per user.
        :param object_cap: (int or str) Cap on number of objects per user. If 'max', leave uncapped.
        :param shot_methods: (str, str) Method for sampling videos for context and target sets.
        :param shots: (int, int) Number of videos to sample for context and target sets.
        :param video_types: (str, str) Video types to sample for context and target sets.
        :param subsample_factor: (int) Factor to subsample video frames before sampling clips.
        :param num_clips: (str, str) Number of clips of contiguous frames to sample from the video. If 'max', sample all non-overlapping clips. If random, sample random number of non-overlapping clips.
        :param clip_length: (int) Number of contiguous frames per video clip.
        :param preload_clips: (bool) If True, preload clips from disk and return as tensors, otherwise return clip paths.
        :param frame_size: (int) Size in pixels of preloaded frames.
        :param annotations_to_load: (list::str) Types of frame annotations to load from disk and return per task.
        :param test_mode: (bool) If True, returns task with target set grouped by video, otherwise returns task with target set not grouped by video.
        :param with_cluster_labels: (bool) If True, use object cluster labels, otherwise use raw object labels.
        :param with_caps: (bool) If True, impose caps on the number of videos per object, otherwise leave uncapped.
        :return: Nothing.
        """
        ORBITDataset.__init__(self, root, way_method, object_cap, shot_methods, shots, video_types, subsample_factor, num_clips, clip_length, preload_clips, frame_size, annotations_to_load, test_mode, with_cluster_labels, with_caps)

    def __getitem__(self, index):
        """
        Function to get a object-centric task as a set of (context and target) clips and labels.
        :param index: (tuple) Task ID and whether to load task target set.
        :return: (dict) Context and target set data for task.
        """
        _, with_target_set = index
        all_objects = range(0, len(self.obj2vids)) # task can consider all possible objects, not just 1 user's objects
        return self.sample_task(all_objects, with_target_set)
