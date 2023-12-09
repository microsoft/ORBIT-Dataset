"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file run_cnaps.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/run_cnaps.py)
from the cambridge-mlg/cnaps library (https://github.com/cambridge-mlg/cnaps).

The original license is included below:

Copyright (c) 2019 John Bronskill, Jonathan Gordon, and James Requeima.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
"""

import os
import time
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from data.dataloaders import DataLoader
from data.utils import unpack_task, attach_frame_history
from model.few_shot_recognisers import MultiStepFewShotRecogniser
from utils.args import parse_args
from utils.optim import cross_entropy
from utils.eval_metrics import TestEvaluator
from utils.logging import print_and_log, get_log_files, stats_to_str

torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    learner = Learner()
    learner.run()

class Learner:
    def __init__(self):
        self.args = parse_args(learner='multi-step-learner')

        self.checkpoint_dir, self.logfile, _, _ \
            = get_log_files(self.args.checkpoint_dir, self.args.model_path)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        device_id = 'cpu'
        if torch.cuda.is_available() and self.args.gpu >= 0:
            cudnn.enabled = True
            cudnn.benchmark = False
            cudnn.deterministic = True
            device_id = 'cuda:' + str(self.args.gpu)
            torch.cuda.manual_seed_all(self.args.seed)

        self.device = torch.device(device_id)
        self.init_dataset()
        self.init_evaluators()
        self.model = self.init_model()
        self.loss = cross_entropy
        
        print_and_log(self.logfile, f"Model details:\n"  \
                f"\tfeature extractor: {self.args.feature_extractor} (pretrained: True, learnable: {self.args.learn_extractor}, finetune film params: {self.args.adapt_features})\n" \
                f"\tclassifier: {self.args.classifier} with logit scale={self.args.logit_scale}\n")

    def init_dataset(self):

        dataset_info = {
            'mode': self.args.mode,
            'data_path': self.args.data_path,
            'test_object_cap': self.args.test_object_cap,
            'test_way_method': self.args.test_way_method,
            'test_shot_methods': [self.args.test_context_shot_method, self.args.test_target_shot_method],
            'num_test_tasks': self.args.num_test_tasks,
            'test_set': self.args.test_set,
            'shots': [self.args.context_shot, self.args.target_shot],
            'video_types': [self.args.context_video_type, self.args.target_video_type],
            'clip_length': self.args.clip_length,
            'test_clip_methods': [self.args.test_context_clip_method, self.args.test_target_clip_method],
            'subsample_factor': self.args.subsample_factor,
            'frame_size': self.args.frame_size,
            'frame_norm_method': self.args.frame_norm_method,
            'annotations_to_load': self.args.annotations_to_load,
            'test_filter_by_annotations': [self.args.test_filter_context, self.args.test_filter_target],
            'logfile': self.logfile
        }

        dataloader = DataLoader(dataset_info)
        self.test_queue = dataloader.get_test_queue()
        
    def init_model(self):
        model = MultiStepFewShotRecogniser(
            self.args.feature_extractor, self.args.adapt_features, self.args.classifier, self.args.clip_length,
            self.args.batch_size, self.args.learn_extractor, self.args.logit_scale
        )
        model._set_device(self.device)
        model._send_to_device()

        return model
   
    def init_finetuner(self):
        finetuner = self.init_model()
        finetuner.load_state_dict(self.model.state_dict(), strict=False)
        finetuner.set_test_mode(True)
        return finetuner

    def init_evaluators(self):
        self.evaluation_metrics = ['frame_acc']

        self.test_evaluator = TestEvaluator(self.evaluation_metrics, self.checkpoint_dir, with_ops_counter=True, count_backwards=True)
    
    def run(self):
        self.test(self.args.model_path)
        self.logfile.close()

    def test(self, path, save_evaluator=True):
        
        self.model = self.init_model()
        if path and os.path.exists(path): # if path exists, load from disk
            self.model.load_state_dict(torch.load(path), strict=False)
        else:
            print_and_log(self.logfile, 'warning: saved model path could not be found; using original param initialisation.')
            path = self.checkpoint_dir
        self.test_evaluator.set_base_params(self.model)
        num_context_clips_per_task, num_target_clips_per_task = [], []
         
        # loop through test tasks (num_test_users * num_test_tasks_per_user)
        num_test_tasks = len(self.test_queue) * self.args.num_test_tasks
        for step, task_dict in enumerate(self.test_queue.get_tasks()):
            context_clips, context_paths, context_labels, target_frames_by_video, target_paths_by_video, target_labels_by_video, object_list = unpack_task(task_dict, self.device, context_to_device=False)
            num_context_clips = len(context_clips)
            self.test_evaluator.set_task_object_list(object_list)
            self.test_evaluator.set_task_context_paths(context_paths)
            
            # initialise finetuner model to initial state of self.model for current task
            finetuner = self.init_finetuner()

            # adapt to current task by finetuning on context clips
            t1 = time.time()
            learning_args= {
                            'num_grad_steps': self.args.personalize_num_grad_steps, 
                            'learning_rate': self.args.personalize_learning_rate,
                            'extractor_lr_scale': self.args.personalize_extractor_lr_scale,
                            'loss_fn': self.loss,
                            'optimizer': self.args.personalize_optimizer,
                            'momentum' : self.args.personalize_momentum,
                            'weight_decay' : self.args.personalize_weight_decay,
                            'betas' : self.args.personalize_betas,
                            'epsilon' : self.args.personalize_epsilon
                            }
            finetuner.personalise(context_clips, context_labels, learning_args, ops_counter=self.test_evaluator.ops_counter)
            self.test_evaluator.log_time(time.time() - t1, 'personalise')

            # loop through target videos for the current task
            with torch.no_grad():
                num_target_clips = 0
                video_iterator = zip(target_frames_by_video, target_paths_by_video, target_labels_by_video)
                for video_frames, video_paths, video_label in video_iterator:
                    video_clips = attach_frame_history(video_frames, self.args.clip_length)
                    num_clips = len(video_clips)
                    t1 = time.time()
                    video_logits = finetuner.predict(video_clips)
                    self.test_evaluator.log_time((time.time() - t1)/float(num_clips), 'inference')
                    self.test_evaluator.append_video(video_logits, video_label, video_paths)
                    num_target_clips += num_clips
                
                # log number of clips per task
                num_context_clips_per_task.append(num_context_clips)
                num_target_clips_per_task.append(num_target_clips)

                # if this is the user's last task, get the average performance for the user over all their tasks
                if (step+1) % self.args.num_test_tasks == 0:
                    self.test_evaluator.set_current_user(task_dict["task_id"])
                    _,_,_,current_video_stats = self.test_evaluator.get_mean_stats(current_user=True)
                    print_and_log(self.logfile, f'{self.args.test_set} user {task_dict["task_id"]} ({self.test_evaluator.current_user+1}/{len(self.test_queue)}) stats: {stats_to_str(current_video_stats)} avg # context clips/task: {np.mean(num_context_clips_per_task):.0f} avg # target clips/task: {np.mean(num_target_clips_per_task):.0f}')
                    if (step+1) < num_test_tasks:
                        num_context_clips_per_task, num_target_clips_per_task = [], [] # reset per user
                        self.test_evaluator.next_user()
                else:
                    self.test_evaluator.next_task()
            
            self.model._reset()

        # get average performance over all users
        stats_per_user, stats_per_obj, stats_per_task, stats_per_video = self.test_evaluator.get_mean_stats()
        stats_per_user_str, stats_per_obj_str, stats_per_task_str, stats_per_video_str = stats_to_str(stats_per_user), stats_to_str(stats_per_obj), stats_to_str(stats_per_task), stats_to_str(stats_per_video)
        mean_ops_stats = self.test_evaluator.get_ops_counter_mean_stats()
        print_and_log(self.logfile, f'{self.args.test_set} [{path}]\n per-user stats: {stats_per_user_str}\n per-object stats: {stats_per_obj_str}\n per-task stats: {stats_per_task_str}\n per-video stats: {stats_per_video_str}\n model stats: {mean_ops_stats}\n')
        if save_evaluator:
            self.test_evaluator.save()
        self.test_evaluator.reset()
    
if __name__ == "__main__":
    main()
