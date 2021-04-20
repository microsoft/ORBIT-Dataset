# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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

    def get_confidence(self, scores):
        return (1.96 * np.std(scores)) / np.sqrt(len(scores))
    
    """
    Function to compute frame accuracy metric.
    :param labels: (np.ndarray) Labels for every frame in a clip (if train) or video (if validation/test).
    :param predictions: (np.ndarray) Predictions for every frame in a clip (if train) or video (if validation/test).
    :return: (float) Average accuracy of a clip (if train) or video (if validation/test).
    """
    def get_frame_accuracy(self, labels, predictions):
        correct_frames = np.equal(labels, predictions).astype(int)
        return np.mean(correct_frames)
    
    """
    Function to compute video accuracy metric.
    :param labels: (np.ndarray) Video-level labels for a list of videos.
    :param predictions: (np.ndarray) Video-level predictions for a list of videos.
    :return: (float) Average accuracy of list of videos.
    """
    def get_video_accuracy(self, labels, predictions):
        correct_videos = np.equal(labels, predictions).astype(int)
        return np.mean(correct_videos)
    
    """
    Function to compute frames-to-recognition metric.
    :param labels: (np.ndarray) Labels for every frame in a video.
    :param predictions: (np.ndarray) Predictions for every frame in a video.
    :return: (float) Number of frames (from first frame) until a correct prediction is made, normalized by video length.
    """
    def get_frames_to_recognition(self, labels, predictions):
        
        correct_frames = np.where(labels == predictions)[0]
        if len(correct_frames) > 0:
            return correct_frames[0] / len(labels) # first correct frame / num_frames
        else: # no frames with correct prediction
            return 1.0
  
    """
    Function to compute the video-level prediction for a video
    :param predictions: (np.ndarray) Predictions for every frame in the video.
    :return: (integer) Most frequent frame prediction in the video.
    """
    def get_video_prediction(self, predictions):
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
        predictions = torch.argmax(test_logits, dim=-1).cpu().numpy()

        for stat in self.stats_to_compute:
            stat_fn = self.frame_stat_fns[stat] if stat in self.frame_stat_fns else self.video_stat_fns[stat]
            self.current_stats[stat] = stat_fn(labels, predictions)
            self.running_stats[stat].append(self.current_stats[stat])
  
    def get_current_stats(self):
        return self.current_stats

    def get_mean_stats(self):
        
        mean_stats = {}
        for stat, score_list in self.running_stats.items():
            mean_score = np.mean(score_list)
            std_dev = np.std(score_list)
            confidence = (1.96 * std_dev) / np.sqrt(len(score_list))
            mean_stats[stat] = [mean_score, confidence]
        
        return mean_stats
    
class TestEvaluator(Evaluator):
    def __init__(self, stats_to_compute):
        super().__init__(stats_to_compute)

        self.current_user = 0
        self.all_frame_predictions, self.all_frame_labels = [[]], [[]]
        self.all_video_predictions, self.all_video_labels = [[]], [[]]
    
    def save(self, path):
        torch.save(self, path + ".test_evaluator")

    def reset(self):
        self.current_user = 0
        self.all_frame_predictions, self.all_frame_labels = [[]], [[]]
        self.all_video_predictions, self.all_video_labels = [[]], [[]]
    
    def get_mean_stats(self, current_user=False):
       
        user_scores = { stat: [] for stat in self.stats_to_compute }
        video_scores = { stat: [] for stat in self.stats_to_compute }
        frame_preds_by_video, frame_labels_by_video, video_preds_by_video, video_labels_by_video = [], [], [], []

        users_to_average = [self.current_user] if current_user else range(self.current_user)
        for stat in self.stats_to_compute:
            for user in users_to_average:
                user_frame_preds = self.all_frame_predictions[user] # [ list_of_video_1_preds, ..., list_of_video_N_preds ]
                user_frame_labels = self.all_frame_labels[user] # [ list_of_video_1_labels, ..., list_of_video_N_labels ]
                user_video_preds = self.all_video_predictions[user] # [ video_1_pred, ..., video_N_preds ]
                user_video_labels = self.all_video_labels[user] # [ video_1_label, ..., video_N_label ] 
                
                # loop over target videos for current user
                if stat in self.frame_stat_fns: # if frame-based metric
                    user_video_scores = [ self.frame_stat_fns[stat](l, p) for (l,p) in zip(user_frame_labels, user_frame_preds) ] # [ video_1_frame_acc, ..., video_N_frame_acc ]
                elif stat in self.video_stat_fns: # if video-based metric 
                    user_video_scores = [ self.video_stat_fns[stat](l, p) for (l,p) in zip(user_video_labels, user_video_preds) ] # [ video_1_video_acc, ..., video_N_video_acc ]

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
            mean_stats[stat] = [ np.mean(user_means), self.get_confidence(user_means) ]

        return mean_stats

    def next_user(self):
        self.current_user += 1
        self.all_frame_labels.append([])
        self.all_frame_predictions.append([])
        self.all_video_labels.append([])
        self.all_video_predictions.append([])
    
    def append(self, test_logits, test_labels):
        frame_labels = test_labels.clone().cpu().numpy()
        frame_predictions = torch.argmax(test_logits, dim=-1).cpu().numpy()
        video_label = frame_labels[0] # all frame labels are the same for a clip/video
        video_prediction = self.get_video_prediction(frame_predictions)

        self.all_frame_labels[self.current_user].append(frame_labels)
        self.all_frame_predictions[self.current_user].append(frame_predictions)
        
        self.all_video_labels[self.current_user].append(video_label)
        self.all_video_predictions[self.current_user].append(video_prediction)

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
