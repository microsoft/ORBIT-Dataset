# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import torch
import numpy as np
from pathlib import Path

class Evaluator():
    def __init__(self, stats_to_compute):

        self.stats_to_compute = stats_to_compute
        self.stat_fns = {
                            'frame_acc' : self.get_frame_accuracy, # frame accuracy
                            'frames_to_recognition' : self.get_frames_to_recognition, # frames to recognition
                            'video_acc' : self.get_video_accuracy, # video accuracy
                        }

    def get_confidence_interval(self, scores):
        return (1.96 * np.std(scores)) / np.sqrt(len(scores))

    def get_frame_accuracy(self, label, predictions):
        """
        Function to compute frame accuracy metric for a clip or video.
        :param label: (np.ndarray) Clip (if train) or video (if validation/test) label.
        :param predictions: (np.ndarray) Predicted probabilities over classes for every frame in a clip (if train) or video (if validation/test).
        :return: (float) Average frame accuracy for clip (if train) or video (if validation/test).
        """
        correct = np.equal(label, predictions).astype(int)
        return np.mean(correct)

    def get_video_accuracy(self, label, predictions):
        """
        Function to compute video accuracy metric for a video.
        :param label: (np.ndarray) Video label.
        :param predictions: (np.ndarray) Predictions over classes for every frame in the video.
        :return: (float) Video accuracy for video.
        """
        most_freq_prediction = self.get_video_prediction(predictions)
        return 1.0 if most_freq_prediction == label else 0.0

    def get_frames_to_recognition(self, label, predictions):
        """
        Function to compute frames-to-recognition metric for a video.
        :param label: (np.ndarray) Labels for every frame in a video.
        :param predictions: (np.ndarray) Predictions over classes for every frame in the video.
        :return: (float) Number of frames (from first frame) until a correct prediction is made, normalized by video length.
        """
        correct = np.where(label == predictions)[0]
        if len(correct) > 0:
            return correct[0] / len(predictions) # first correct frame index / num_frames
        else:
            return 1.0 # no correct predictions; last frame index / num_frames (i.e. 1.0)

    def get_video_prediction(self, predictions):
        """
        Function to compute the video-level prediction for a video
        :param predictions: (np.ndarray) Predictions over classes for every frame in the video.
        :return: (integer) Most frequent frame prediction in the video.
        """
        return np.bincount(predictions).argmax()

class TrainEvaluator(Evaluator):
    def __init__(self, stats_to_compute):
        super().__init__(stats_to_compute)
        self.current_stats = { stat: 0.0 for stat in self.stats_to_compute }
        self.running_stats = { stat: [] for stat in self.stats_to_compute }

    def reset(self):
        self.current_stats = { stat: 0.0 for stat in self.stats_to_compute }
        self.running_stats = { stat: [] for stat in self.stats_to_compute }

    def update_stats(self, logits, labels):
        labels = labels.clone().cpu().numpy()
        predictions = logits.argmax(dim=-1).detach().cpu().numpy()

        for stat in self.stats_to_compute:
            self.current_stats[stat] = self.stat_fns[stat](labels, predictions)
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
    def __init__(self, stats_to_compute, save_dir = None):
        super().__init__(stats_to_compute)
        if save_dir:
            self.save_dir = save_dir
        self.reset()

    def save(self):
        output = {}
        num_users = self.current_user+1
        assert len(self.user_object_lists) == num_users
        for user in range(num_users): # loop through users
            user_frame_paths = self.all_frame_paths[user]
            user_frame_predictions = self.all_frame_predictions[user]
            user_object_list = self.user_object_lists[user]
            user_id = self.user2userid[user]
            output[user_id] = {'user_objects': user_object_list, 'user_videos': {}}

            assert len(user_frame_paths) == len(user_frame_predictions)
            num_videos = len(user_frame_paths)

            for v in range(num_videos): # loop through videos
                video_frame_paths = user_frame_paths[v]
                video_frame_predictions = user_frame_predictions[v].tolist()

                assert len(video_frame_paths) == len(video_frame_predictions)
                video_id = Path(video_frame_paths[0]).parts[-2]
                output[user_id]['user_videos'][video_id] = []

                for i, (path, pred) in enumerate(zip(video_frame_paths, video_frame_predictions)): # loop through frames
                    frame_id = int(Path(path).stem.split('-')[-1])
                    assert frame_id == i+1 # frames must be sorted
                    output[user_id]['user_videos'][video_id].append(pred)

        self.json_results_path = Path(self.save_dir, "results.json")
        self.json_results_path.parent.mkdir(exist_ok=True, parents=True)
        with open(self.json_results_path, 'w') as json_file:
            json.dump(output, json_file)

    def reset(self):
        self.current_user = 0
        self.all_frame_predictions = [[]]
        self.all_video_labels = [[]]
        self.all_frame_paths = [[]]
        self.user_object_lists = []
        self.user2userid = {}

    def get_mean_stats(self, current_user=False):
        user_scores = { stat: [] for stat in self.stats_to_compute }
        video_scores = { stat: [] for stat in self.stats_to_compute }

        num_users = self.current_user+1
        users_to_average = [self.current_user] if current_user else range(num_users)
        for stat in self.stats_to_compute:
            for user in users_to_average:
                user_frame_preds = self.all_frame_predictions[user] # [ video_1_frame_predss, ..., video_N_frame_preds ]
                user_video_labels = self.all_video_labels[user] # [ video_1_label, ..., video_N_label ]

                # loop over target videos for current user
                user_video_scores = []
                for video_label, frame_preds in zip(user_video_labels, user_frame_preds):
                    video_score = self.stat_fns[stat](video_label, frame_preds)
                    user_video_scores.append(video_score)

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
        self.all_frame_predictions.append([])
        self.all_video_labels.append([])
        self.all_frame_paths.append([])

    def append_video(self, frame_logits, video_label, frame_paths, object_list):

        # remove any duplicate frames added due to padding to a multiple of clip_length
        frame_paths, unique_idxs = np.unique(frame_paths, return_index=True)
        frame_logits = frame_logits[unique_idxs]

        assert frame_paths.shape[0] == frame_logits.shape[0]

        frame_predictions = frame_logits.argmax(dim=-1).detach().cpu().numpy()
        video_label = video_label.clone().cpu().numpy()

        # append results to current user to log
        self.all_frame_predictions[self.current_user].append(frame_predictions)
        self.all_video_labels[self.current_user].append(video_label)
        self.all_frame_paths[self.current_user].append(frame_paths)
        if self.current_user not in self.user2userid: # if a new user, log them (and their objects)
            self.user_object_lists.append(object_list)
            user_id = Path(frame_paths[0]).stem.split('--')[0]
            self.user2userid[self.current_user] = user_id

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
