# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import math
import numpy as np

class Evaluator():
    def __init__(self, stats_to_compute):

        self.stats_to_compute = stats_to_compute
        self.frame_stat_fns = {
                                'frame_acc' : self.get_frame_accuracy, # frame accuracy
                                'frames_to_recognition' : self.get_frames_to_recognition, # frames to recognition
                              }
        self.video_stat_fns = {
                                'video_acc' : self.get_video_accuracy, # video accuracy
                              }

    def get_confidence_interval(self, scores):
        return (1.96 * np.std(scores)) / np.sqrt(len(scores))
    
    """
    Function to compute frame accuracy metric.
    :param labels: (np.ndarray) Labels for every frame in a clip (if train) or video (if validation/test).
    :param probs: (np.ndarray) Predicted probabilities over classes for every frame in a clip (if train) or video (if validation/test).
    :return: (float) Average accuracy of a clip (if train) or video (if validation/test).
    """
    def get_frame_accuracy(self, labels, probs):
        predictions = np.argmax(probs, axis=-1)
        correct = np.equal(labels, predictions).astype(int)
        return np.mean(correct)
    
    """
    Function to compute video accuracy metric.
    :param labels: (np.ndarray) Video-level labels for a list of videos.
    :param probs: (np.ndarray) Predicted probabilities over classes for every frame in a video.
    :return: (float) Average accuracy of list of videos.
    """
    def get_video_accuracy(self, labels, probs):
        predictions = self.get_video_prediction(probs)
        correct_videos = np.equal(labels, predictions).astype(int)
        return np.mean(correct_videos)
    
    """
    Function to compute frames-to-recognition metric.
    :param labels: (np.ndarray) Labels for every frame in a video.
    :param probs: (np.ndarray) Predicted probabilities over classes for every frame in a video.
    :return: (float) Number of frames (from first frame) until a correct prediction is made, normalized by video length.
    """
    def get_frames_to_recognition(self, labels, probs):
       
        predictions = np.argmax(probs, axis=-1)
        correct = np.where(labels == predictions)[0]
        if len(correct) > 0:
            return correct[0] / len(labels) # first correct frame / num_frames
        else: # no frames with correct prediction
            return 1.0
     
    """
    Function to compute the video-level prediction for a video
    :param probs: (np.ndarray) Predicted probabilities over classes for every frame in the video.
    :return: (integer) Most frequent frame prediction in the video.
    """
    def get_video_prediction(self, probs):
        predictions = np.argmax(probs, axis=-1)
        return np.bincount(predictions).argmax()

class TrainEvaluator(Evaluator):
    def __init__(self, stats_to_compute):
        super().__init__(stats_to_compute)
        self.current_stats = { stat: 0.0 for stat in self.stats_to_compute }
        self.running_stats = { stat: [] for stat in self.stats_to_compute }
    
    def reset(self):
        self.current_stats = { stat: 0.0 for stat in self.stats_to_compute }
        self.running_stats = { stat: [] for stat in self.stats_to_compute }
    
    def update_stats(self, test_logits, test_labels):
        labels = test_labels.clone().cpu().numpy()
        probs = torch.nn.functional.softmax(test_logits, dim=-1).detach().cpu().numpy()
        for stat in self.stats_to_compute:
            stat_fn = self.frame_stat_fns[stat] if stat in self.frame_stat_fns else self.video_stat_fns[stat]
            self.current_stats[stat] = stat_fn(labels, probs)
            self.running_stats[stat].append(self.current_stats[stat])
  
    def get_current_stats(self):
        return self.current_stats

    def get_mean_stats(self):
        
        mean_stats = {}
        for stat, score_list in self.running_stats.items():
            mean_score = np.mean(score_list)
            confidence_interval = self.get_confidence_interval(score_list)
            mean_stats[stat] = [mean_score, confidence_interval]
        
        return mean_stats
    
class TestEvaluator(Evaluator):
    def __init__(self, stats_to_compute):
        super().__init__(stats_to_compute)

        self.current_user = 0
        self.all_frame_probs, self.all_frame_labels = [[]], [[]]
    
    def save(self, path):
        if os.path.isfile(path):
            torch.save(self, path + ".test_evaluator")
        else:
            torch.save(self, os.path.join(path, "test_evaluator"))

    def reset(self):
        self.current_user = 0
        self.all_frame_probs, self.all_frame_labels = [[]], [[]]
    
    def get_mean_stats(self, current_user=False):
        
        user_scores = { stat: [] for stat in self.stats_to_compute }
        video_scores = { stat: [] for stat in self.stats_to_compute }
        
        users_to_average = [self.current_user] if current_user else range(self.current_user)
        for stat in self.stats_to_compute:
            for user in users_to_average:
                user_frame_probs = self.all_frame_probs[user] # [ list_of_video_1_probs, ..., list_of_video_N_probs ]
                user_frame_labels = self.all_frame_labels[user] # [ list_of_video_1_labels, ..., list_of_video_N_labels ]
                user_video_labels = [vl[0] for vl in self.all_frame_labels[user]] # [ video_1_label, ..., video_N_label ], first label in frame_labels ==  video label
                
                # loop over target videos for current user
                if stat in self.frame_stat_fns: # if frame-based metric
                    user_video_scores = [ self.frame_stat_fns[stat](l, p) for (l,p) in zip(user_frame_labels, user_frame_probs) ] # [ video_1_frame_acc, ..., video_N_frame_acc ]
                elif stat in self.video_stat_fns: # if video-based metric 
                    user_video_scores = [ self.video_stat_fns[stat](l, p) for (l,p) in zip(user_video_labels, user_frame_probs) ] # [ video_1_video_acc, ..., video_N_video_acc ]
                elif stat in self.uncertainty_stat_fns: # if uncertainty-based metric
                    user_video_scores = [ self.uncertainty_stat_fns[stat](l, p) for (l,p) in zip(user_frame_labels, user_frame_probs) ] # [ video_1_uncertainty, ..., video_N_uncertainty ]

                user_mean = np.mean(user_video_scores) # compute mean over target videos for current user
                user_scores[stat].append( user_mean ) # append mean over target videos for current user
                video_scores[stat].extend( user_video_scores ) # accumulate list of video scores

        # computes average score over all users
        user_stats = self.average_over_scores(user_scores) # user_scores: [user_1_mean, ..., user_M_mean]
        # computes average score over all videos (pooled across users)
        video_stats = self.average_over_scores(video_scores) # video_scores: [user_1_video_1_mean, user_1_video_2_mean, ..., user_M_video_N_mean]
        return user_stats, video_stats
       
    def average_over_scores(self, user_stats):

        mean_stats = {}
        for stat in self.stats_to_compute:
            user_means = user_stats[stat]
            mean_stats[stat] = [ np.mean(user_means), self.get_confidence_interval(user_means) ]

        return mean_stats

    def next_user(self):
        self.current_user += 1
        self.all_frame_labels.append([])
        self.all_frame_probs.append([])
    
    def append(self, test_logits, test_labels):

        frame_labels = test_labels.clone().cpu().numpy()
        frame_probs = torch.nn.functional.softmax(test_logits, dim=-1).detach().cpu().numpy()

        self.all_frame_labels[self.current_user].append(frame_labels)
        self.all_frame_probs[self.current_user].append(frame_probs) 
    
class ValidationEvaluator(TestEvaluator):
    def __init__(self, stats_to_compute):
        super().__init__(stats_to_compute)
        self.comparison_stat = self.stats_to_compute[0] # first stat is used to validate model
        self.current_best_stats = { stat : [0.0, 0.0] for stat in self.stats_to_compute }
    
    def is_better(self, stats):
        is_better = False
        # compare stats with current_best_stats
        if stats[self.comparison_stat][0] > self.current_best_stats[self.comparison_stat][0]:
            is_better = True

        return is_better

    def replace(self, stats):
        self.current_best_stats = stats

    def get_current_best_stats(self):
        return self.current_best_stats
