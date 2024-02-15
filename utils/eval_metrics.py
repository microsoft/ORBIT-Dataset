# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import torch
import numpy as np
from pathlib import Path
from thop import clever_format

from utils.ops_counter import OpsCounter
from utils.logging import convert_to_minutes, convert_to_microseconds

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

    def get_frame_accuracy(self, label, probs):
        """
        Function to compute frame accuracy metric for a clip or video.
        :param label: (np.ndarray) Clip or video label.
        :param probs: (np.ndarray) Predicted probabilities over classes for every frame in the clip/video.
        :return: (float) Average frame accuracy for clip (if train) or video (if validation/test).
        """
        predictions = np.argmax(probs, axis=-1)
        correct = np.equal(label, predictions).astype(int)
        return np.mean(correct)

    def get_video_accuracy(self, label, probs):
        """
        Function to compute video accuracy metric for a video.
        :param label: (np.ndarray) Video label.
        :param probs: (np.ndarray) Predicted probabilities over classes for every frame in a video.
        :return: (float) Video accuracy for video.
        """
        most_freq_prediction = self.get_video_prediction(probs)
        return 1.0 if most_freq_prediction == label else 0.0

    def get_frames_to_recognition(self, label, probs):
        """
        Function to compute frames-to-recognition metric for a video.
        :param label: (np.ndarray) Labels for every frame in a video.
        :param probs: (np.ndarray) Predicted probabilities over classes for every frame in a video.
        :return: (float) Number of frames (from first frame) until a correct prediction is made, normalized by video length.
        """
        predictions = np.argmax(probs, axis=-1)
        correct = np.where(label == predictions)[0]
        if len(correct) > 0:
            return correct[0] / len(predictions) # first correct frame index / num_frames
        else:
            return 1.0 # no correct predictions; last frame index / num_frames (i.e. 1.0)

    def get_video_prediction(self, probs):
        """
        Function to compute the video-level prediction for a video
        :param probs: (np.ndarray) Predicted probabilities over classes for every frame in the video.
        :return: (integer) Most frequent frame prediction in the video.
        """
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

    def update_stats(self, logits, labels):
        labels = labels.clone().cpu().numpy()
        probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()

        for stat in self.stats_to_compute:
            self.current_stats[stat] = self.stat_fns[stat](labels, probs)
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
    def __init__(self, stats_to_compute, save_dir = None, with_ops_counter = False, count_backwards = False):
        super().__init__(stats_to_compute)
        if save_dir:
            self.save_dir = save_dir
        if with_ops_counter:
            self.ops_counter = OpsCounter(count_backward=count_backwards)
        else:
            self.ops_counter = None
        self.reset()

    def save(self):
        output = {}
        num_users = self.current_user+1

        assert len(self.all_users) == num_users
        for user in range(num_users): # loop through users
            user_id = self.all_users[user]
            user_frame_paths = self.all_frame_paths[user]
            user_frame_probs = self.all_frame_probs[user]
            user_frame_predictions = self.all_frame_predictions[user]
            user_object_lists = self.all_object_lists[user]
            num_tasks = len(user_frame_paths)
            output[user_id] = []
            for task in range(num_tasks): # loop through tasks per user
                task_frame_paths = user_frame_paths[task]
                task_frame_probs = user_frame_probs[task]
                task_frame_predictions = user_frame_predictions[task]
                task_object_list = user_object_lists[task]
                num_videos = len(task_frame_paths)

                task_output = {'task_object_list': task_object_list, 'task_videos': {}}
                if self.ops_counter:
                    task_output['task_macs_to_personalise'] = int(self.macs_counter[user][task])
                for v in range(num_videos): # loop through videos per task
                    video_frame_paths = task_frame_paths[v]
                    video_frame_probs = task_frame_probs[v].tolist()
                    video_frame_predictions = task_frame_predictions[v]

                    assert len(video_frame_paths) == len(video_frame_probs) == len(video_frame_predictions)
                    video_id = Path(video_frame_paths[0]).parts[-2]
                    task_output['task_videos'][video_id] = {}

                    for i, (path, probs, pred) in enumerate(zip(video_frame_paths, video_frame_probs, video_frame_predictions)): # loop through frames
                        frame_id = int(Path(path).stem.split('-')[-1])
                        task_output['task_videos'][video_id].update( {frame_id: pred} )

                output[user_id].append(task_output)

        self.json_results_path = Path(self.save_dir, "results.json")
        self.json_results_path.parent.mkdir(exist_ok=True, parents=True)
        with open(self.json_results_path, 'w') as json_file:
            json.dump(output, json_file)

    def get_mean_stats(self, current_user=False):
        user_scores = { stat: [] for stat in self.stats_to_compute }
        video_scores = { stat: [] for stat in self.stats_to_compute }
        task_scores = { stat: [] for stat in self.stats_to_compute }
        object_scores = { stat: [] for stat in self.stats_to_compute }

        num_users = self.current_user+1
        users_to_average = [self.current_user] if current_user else range(num_users)
        for stat in self.stats_to_compute:
            for user in users_to_average:
                user_frame_probs = self.all_frame_probs[user] # [ task_1_frame_probs, ..., task_N_frame_probs ]
                user_video_labels = self.all_video_labels[user] # [ task_1_labels, ..., task_N_labels ]

                obj2flatframeprobs = {}
                num_tasks = len(user_frame_probs)
                flat_user_frame_probs, flat_user_frame_labels = [], []
                for task in range(num_tasks): # loop over all tasks for current user
                    task_frame_probs = user_frame_probs[task]
                    task_video_labels = user_video_labels[task]

                    # loop over target videos for current task
                    flat_task_frame_probs, flat_task_frame_labels = [], []
                    for video_label, frame_probs in zip(task_video_labels, task_frame_probs):
                        video_score = self.stat_fns[stat](video_label, frame_probs)
                        video_scores[stat].append(video_score) # accumulate list of scores per video

                        flat_task_frame_probs.extend(frame_probs)
                        flat_task_frame_labels.extend(video_label.repeat(frame_probs.shape[0]))

                        object_label = int(video_label)
                        if object_label in list(obj2flatframeprobs.keys()):
                            obj2flatframeprobs[object_label].extend(frame_probs)
                        else:
                            obj2flatframeprobs[object_label] = list(frame_probs)

                    task_score = self.stat_fns[stat](np.array(flat_task_frame_labels), np.row_stack(flat_task_frame_probs))
                    task_scores[stat].append(task_score) # accumulate list of scores per task

                    flat_user_frame_probs.extend(flat_task_frame_probs)
                    flat_user_frame_labels.extend(flat_task_frame_labels)

                for obj, flat_obj_frame_probs in obj2flatframeprobs.items():
                    obj_score = self.stat_fns[stat](np.array(obj), np.row_stack(flat_obj_frame_probs))
                    object_scores[stat].append(obj_score)

                user_score = self.stat_fns[stat](np.array(flat_user_frame_labels), np.row_stack(flat_user_frame_probs))
                user_scores[stat].append(user_score)

        # computes average score over all users
        user_stats = self.average_over_scores(user_scores) # user_scores: [user_1_mean, ..., user_M_mean]
        # computes average score over all objects
        object_stats = self.average_over_scores(object_scores) # object_scores: [user_1_object_1_mean, user_1_object_2_mean, ..., user_M_object_K_mean]
        # computes average score over all tasks
        task_stats = self.average_over_scores(task_scores) # task_scores: [user_1_task_1_mean, user_1_task_2_mean, ..., user_M_task_T_mean]
        # computes average score over all videos
        video_stats = self.average_over_scores(video_scores) # video_scores: [user_1_video_1_mean, user_1_video_2_mean, ..., user_M_video_N_mean]
        return user_stats, object_stats, task_stats, video_stats

    def average_over_scores(self, user_stats):
        mean_stats = {}
        for stat in self.stats_to_compute:
            user_means = user_stats[stat]
            mean_stats[stat] = [ np.mean(user_means), self.get_confidence_interval(user_means) ]

        return mean_stats

    def get_mean_ops_counter_stats(self, current_user=False):
        if self.ops_counter:
            num_users = self.current_user+1
            users_to_average = [self.current_user] if current_user else range(num_users)
            task_macs = [tm for user in users_to_average for tm in self.macs_counter[user]] # unravel MACs per task into a flat list of floats
            task_params = [tp for user in users_to_average for tp in self.params_counter[user]] # unravel params per task into a flat list of floats
            mean_ops = np.mean(task_macs)
            std_ops = np.std(task_macs)
            mean_params = np.mean(task_params)

            mean_ops, std_ops, mean_params = clever_format([mean_ops, std_ops, mean_params], "%.2f")
            return mean_ops, std_ops, mean_params, self.ops_counter.params_break_down
        else:
            return "0.00B", "0.00B", "0.00B", ""
    
    def get_mean_times(self, current_user=False):
        num_users = self.current_user+1
        users_to_average = [self.current_user] if current_user else range(num_users)
        personalise_times_per_task = []
        inference_times_per_task = []
        for user in users_to_average:
            personalise_times_per_task.append(np.mean(self.all_personalise_times[user]))
            inference_times_per_task.append(np.mean(self.all_inference_times[user]))
        
        mean_personalise_time_per_task = convert_to_minutes(np.mean(personalise_times_per_task))
        std_personalise_time_per_task = convert_to_minutes(np.std(personalise_times_per_task))
        mean_inference_time_per_task = convert_to_microseconds(np.mean(inference_times_per_task))
        std_inference_time_per_task = convert_to_microseconds(np.std(inference_times_per_task))

        return mean_personalise_time_per_task, std_personalise_time_per_task, mean_inference_time_per_task, std_inference_time_per_task
    
    def check_for_uncounted_modules(self, model: torch.nn.Module) -> str:
        if self.ops_counter:
            uncounted_mods = "\n".join(self.ops_counter.get_uncounted_modules(model))
            return f"MACs from these modules will not be counted by default.\n \
                If they involve MACs, you will need to write a custom_ops function for\n \
                    each in set_custom_ops() in utils/ops_counter.py: \n{uncounted_mods}"
        return "TestEvaluator has no ops_counter - cannot check if MACs of all modules will be counted."

    def append_video(self, frame_logits, video_label, frame_paths):

        # remove any duplicate frames added due to padding to a multiple of clip_length
        frame_paths, unique_idxs = np.unique(frame_paths, return_index=True)
        frame_logits = frame_logits[unique_idxs]

        assert frame_paths.shape[0] == frame_logits.shape[0]

        frame_probs = torch.nn.functional.softmax(frame_logits, dim=-1).detach().cpu().numpy()
        video_label = video_label.clone().cpu().numpy()
        frame_predictions = frame_logits.argmax(dim=-1).detach().cpu().numpy().tolist()

        # append results to current user to log
        self.all_frame_probs[self.current_user][self.current_task].append(frame_probs)
        self.all_video_labels[self.current_user][self.current_task].append(video_label)
        self.all_frame_paths[self.current_user][self.current_task].append(frame_paths)
        self.all_frame_predictions[self.current_user][self.current_task].append(frame_predictions)

    def reset(self):
        self.current_user = 0
        self.current_task = 0
        self.all_frame_probs = [[[]]]
        self.all_video_labels = [[[]]]
        self.all_frame_paths = [[[]]]
        self.all_frame_predictions = [[[]]]
        self.all_users = []
        self.all_object_lists = [[[]]]
        self.all_personalise_times = [[[]]]
        self.all_inference_times = [[[]]]
        if self.ops_counter:
            self.macs_counter = [[[]]]
            self.params_counter = [[[]]]

    def set_current_user(self, user_id):
        self.all_users.append(user_id)
        assert len(self.all_users) == self.current_user+1

    def set_task_object_list(self, task_object_list):
        self.all_object_lists[self.current_user][self.current_task] = task_object_list

    def next_user(self):
        self.all_frame_probs.append([[]])
        self.all_video_labels.append([[]])
        self.all_frame_paths.append([[]])
        self.all_frame_predictions.append([[]])
        self.all_object_lists.append([[]])
        self.all_personalise_times.append([[]])
        self.all_inference_times.append([[]])
        if self.ops_counter:
            self.macs_counter.append([[]])
            self.params_counter.append([[]])
        self.current_task = 0
        self.current_user += 1

    def next_task(self):
        self.all_frame_probs[self.current_user].append([])
        self.all_video_labels[self.current_user].append([])
        self.all_frame_paths[self.current_user].append([])
        self.all_frame_predictions[self.current_user].append([])
        self.all_object_lists[self.current_user].append([])
        self.all_personalise_times[self.current_user].append([])
        self.all_inference_times[self.current_user].append([])
        if self.ops_counter:
            self.macs_counter[self.current_user].append([])
            self.params_counter[self.current_user].append([])
        self.current_task += 1

    def set_base_params(self, params):
        if self.ops_counter:
            self.ops_counter.set_base_params(params)

    def log_time(self, time: float , time_type:str='personalise'):
        if time_type == 'personalise':
            self.all_personalise_times[self.current_user][self.current_task] = time
        elif time_type == 'inference':
            self.all_inference_times[self.current_user][self.current_task] = time
        else:
            raise ValueError(f"time_type must be 'personalise' or 'inference' but got {time_type}")

    def task_complete(self):
        if self.ops_counter:
            self.macs_counter[self.current_user][self.current_task] = self.ops_counter.get_task_macs()
            self.params_counter[self.current_user][self.current_task] = self.ops_counter.get_task_params()
            self.ops_counter.task_complete()

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
