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
from models.few_shot_recognisers import MultiStepFewShotRecogniser
from utils.args import parse_args
from utils.ops_counter import OpsCounter
from utils.optim import cross_entropy, init_optimizer
from utils.data import unpack_task, attach_frame_history
from utils.logging import print_and_log, get_log_files, stats_to_str
from utils.eval_metrics import TrainEvaluator, ValidationEvaluator, TestEvaluator

torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    learner = Learner()
    learner.run()

class Learner:
    def __init__(self):
        self.args = parse_args(learner='gradient-learner')

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.model_path)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        device_id = 'cpu'
        self.map_location = 'cpu'
        if torch.cuda.is_available() and self.args.gpu >= 0:
            cudnn.enabled = True
            cudnn.benchmark = False
            cudnn.deterministic = True
            device_id = 'cuda:' + str(self.args.gpu)
            torch.cuda.manual_seed_all(self.args.seed)
            self.map_location = lambda storage, loc: storage.cuda()

        self.device = torch.device(device_id)
        self.ops_counter = OpsCounter(count_backward=True)
        self.init_dataset()
        self.init_evaluators()
        self.model = self.init_model()
        self.loss = cross_entropy

    def init_dataset(self):

        dataset_info = {
            'mode': self.args.mode,
            'data_path': self.args.data_path,
            'train_object_cap': self.args.train_object_cap,
            'with_train_shot_caps': self.args.with_train_shot_caps,
            'with_cluster_labels': True,
            'train_way_method': self.args.train_way_method,
            'test_way_method': self.args.test_way_method,
            'train_shot_methods': [self.args.train_context_shot_method, self.args.train_target_shot_method],
            'test_shot_methods': [self.args.test_context_shot_method, self.args.test_target_shot_method],
            'train_tasks_per_user': self.args.train_tasks_per_user,
            'test_tasks_per_user': self.args.test_tasks_per_user,
            'train_task_type': self.args.train_task_type,
            'test_set': self.args.test_set,
            'shots': [self.args.context_shot, self.args.target_shot],
            'video_types': [self.args.context_video_type, self.args.target_video_type],
            'clip_length': self.args.clip_length,
            'train_num_clips': [self.args.train_context_num_clips, self.args.train_target_num_clips],
            'test_num_clips': [self.args.test_context_num_clips, self.args.test_target_num_clips],
            'subsample_factor': self.args.subsample_factor,
            'frame_size': self.args.frame_size,
            'annotations_to_load': self.args.annotations_to_load,
            'preload_clips': self.args.preload_clips,
        }

        dataloader = DataLoader(dataset_info)
        self.train_queue = dataloader.get_train_queue()
        self.validation_queue = dataloader.get_validation_queue()
        self.test_queue = dataloader.get_test_queue()
        
    def init_model(self):
        model = MultiStepFewShotRecogniser(
                    self.args.pretrained_extractor_path, self.args.feature_extractor, self.args.batch_normalisation,
                    self.args.adapt_features, self.args.classifier, self.args.clip_length, self.args.batch_size,
                    self.args.learn_extractor, self.args.feature_adaptation_method, self.args.use_two_gpus, self.args.num_grad_steps
                )
        model._register_extra_parameters()
        model._set_device(self.device)
        model._send_to_device()

        return model
   
    def init_finetuner(self):
        finetuner = self.init_model()
        finetuner.load_state_dict(self.model.state_dict(), strict=False)
        finetuner._freeze_extractor() 
        finetuner.set_test_mode(True)
        return finetuner

    def init_evaluators(self):
        self.train_metrics = ['frame_acc']
        self.evaluation_metrics = ['frame_acc', 'frames_to_recognition', 'video_acc']
        self.train_evaluator = TrainEvaluator(self.train_metrics)
        self.validation_evaluator = ValidationEvaluator(self.evaluation_metrics)
        self.test_evaluator = TestEvaluator(self.evaluation_metrics, self.checkpoint_dir)
    
    def run(self):
        if self.args.mode == 'train' or self.args.mode == 'train_test':
         
            num_classes = len(self.train_queue.get_cluster_classes())
            self.model.configure_classifier(num_classes, self.device)
            self.optimizer = init_optimizer(self.model, self.args.learning_rate, extractor_scale_factor=0.1)
            
            for epoch in range(self.args.epochs):
                losses = []
                since = time.time()
                torch.set_grad_enabled(True)
                self.model.set_test_mode(False)

                train_tasks = self.train_queue.get_tasks()
                total_steps = len(train_tasks)
                for step, task_dict in enumerate(train_tasks):

                    t1 = time.time()
                    task_loss = self.train_task(task_dict)
                    task_time = time.time() - t1
                    losses.append(task_loss.detach())

                    if ((step + 1) % self.args.tasks_per_batch == 0) or (step == (total_steps - 1)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if self.args.print_by_step:
                        current_stats_str = stats_to_str(self.train_evaluator.get_current_stats())
                        print_and_log(self.logfile, f'epoch [{epoch+1}/{self.args.epochs}][{step+1}/{total_steps}], train loss: {task_loss.item():.7f}, {current_stats_str.strip()}, time/task: {int(task_time/60):d}m{int(task_time%60):02d}s')

                mean_stats = self.train_evaluator.get_mean_stats()
                mean_epoch_loss = torch.Tensor(losses).mean().item()
                seconds = time.time() - since
                # print
                print_and_log(self.logfile, '-' * 100)
                print_and_log(self.logfile, f'epoch [{epoch+1}/{self.args.epochs}] train loss: {mean_epoch_loss:.7f} {stats_to_str(mean_stats)} time/epoch: {int(seconds/60):d}m{int(seconds%60):02d}s')
                print_and_log(self.logfile, '-' * 100)
                self.train_evaluator.reset()
                self.save_checkpoint(epoch + 1)

                # validate
                if (epoch + 1) >= self.args.validation_on_epoch:
                    self.validate()

            # save the final model
            torch.save(self.model.state_dict(), self.checkpoint_path_final)

        if self.args.mode == 'train_test':
            self.test(self.checkpoint_path_final)
            self.test(self.checkpoint_path_validation)

        if self.args.mode == 'test':
            self.test(self.args.model_path)

        self.logfile.close()

    def train_task(self, task_dict):

        context_clips, context_frames, context_labels, target_clips, target_frames, target_labels, object_list = unpack_task(task_dict, self.device, target_to_device=True, preload_clips=self.args.preload_clips)
      
        joint_context_clips = torch.cat((context_clips, target_clips)) if self.args.preload_clips else np.concatenate((context_clips, target_clips))
        joint_context_labels = torch.cat((context_labels, target_labels), dim=0)
        joint_context_logits = self.model.predict(joint_context_clips, context=True)
        self.train_evaluator.update_stats(joint_context_logits, joint_context_labels) 

        task_loss = self.loss(joint_context_logits, joint_context_labels) / self.args.tasks_per_batch
        task_loss += 0.001 * self.model.feature_adapter.regularization_term(switch_device=self.args.use_two_gpus)
        task_loss.backward()

        return task_loss

    def validate(self):
 
        # loop through validation tasks (num_validation_users * num_test_tasks_per_user)
        num_val_tasks = self.validation_queue.num_users * self.args.test_tasks_per_user
        for step, task_dict in enumerate(self.validation_queue.get_tasks()):
            context_clips, context_paths, context_labels, target_frames_by_video, target_paths_by_video, target_labels_by_video, object_list = unpack_task(task_dict, self.device, context_to_device=False, preload_clips=self.args.preload_clips)
            
            # if this is a user's first task, cache their target videos (as they remain constant for all their tasks - ie. num_test_tasks_per_user)
            if step % self.args.test_tasks_per_user == 0:
                cached_target_frames_by_video, cached_target_paths_by_video, cached_target_labels_by_video = target_frames_by_video, target_paths_by_video, target_labels_by_video

            # initialise finetuner model to initial state of self.model for current task
            finetuner = self.init_finetuner()

            # adapt to current task by finetuning on context clips
            learning_args=(self.args.inner_learning_rate, self.loss, 'sgd', 1.0)
            finetuner.personalise(context_clips, context_labels, learning_args)

            with torch.no_grad():
                # loop through cached target videos for current task
                for video_frames, video_paths, video_label in zip(cached_target_frames_by_video, cached_target_paths_by_video, cached_target_labels_by_video):
                    video_clips = attach_frame_history(video_frames, self.args.clip_length)
                    video_logits = finetuner.predict(video_clips)
                    self.validation_evaluator.append_video(video_logits, video_label, video_paths, object_list)

                # if this is a user's last task, get the average performance for the user
                if (step+1) % self.args.test_tasks_per_user == 0:
                    _, current_user_stats = self.validation_evaluator.get_mean_stats(current_user=True)
                    print_and_log(self.logfile, f'validation user {task_dict["user_id"]} ({self.validation_evaluator.current_user+1}/{self.validation_queue.num_users}) stats: {stats_to_str(current_user_stats)}')
                    if (step+1) < num_val_tasks:
                        self.validation_evaluator.next_user()

        # get average performance over all users
        stats_per_user, stats_per_video = self.validation_evaluator.get_mean_stats()
        stats_per_user_str, stats_per_video_str = stats_to_str(stats_per_user), stats_to_str(stats_per_video)
        print_and_log(self.logfile, f'validation\n per-user stats: {stats_per_user_str}\n per-video stats: {stats_per_video_str}\n')
        # save the model if validation is the best so far
        if self.validation_evaluator.is_better(stats_per_video):
            self.validation_evaluator.replace(stats_per_video)
            torch.save(self.model.state_dict(), self.checkpoint_path_validation)
            print_and_log(self.logfile, 'best validation model was updated.\n')

        self.validation_evaluator.reset()

    def test(self, path):
        
        self.model = self.init_model()
        if path: # if path is None, use model as initialised in init_model()
            self.model.load_state_dict(torch.load(path, map_location=self.map_location), strict=False)
        self.ops_counter.set_base_params(self.model)
         
        # loop through test tasks (num_test_users * num_test_tasks_per_user)
        num_test_tasks = self.test_queue.num_users * self.args.test_tasks_per_user
        for step, task_dict in enumerate(self.test_queue.get_tasks()):
            context_clips, context_paths, context_labels, target_frames_by_video, target_paths_by_video, target_labels_by_video, object_list = unpack_task(task_dict, self.device, context_to_device=False, preload_clips=self.args.preload_clips)
            
            # if this is a user's first task, cache their target videos (as they remain constant for all their tasks - ie. num_test_tasks_per_user)
            if step % self.args.test_tasks_per_user == 0:
                cached_target_frames_by_video, cached_target_paths_by_video, cached_target_labels_by_video = target_frames_by_video, target_paths_by_video, target_labels_by_video

            # initialise finetuner model to initial state of self.model for current task
            finetuner = self.init_finetuner()

            # adapt to current task by finetuning on context clips
            t1 = time.time()
            learning_args=(self.args.inner_learning_rate, self.loss, 'sgd', 1.0)
            finetuner.personalise(context_clips, context_labels, learning_args, ops_counter=self.ops_counter)
            self.ops_counter.log_time(time.time() - t1)
            # add task's ops to self.ops_counter
            self.ops_counter.task_complete()

            # loop through cached videos for the current task
            with torch.no_grad():
                for video_frames, video_paths, video_label in zip(cached_target_frames_by_video, cached_target_paths_by_video, cached_target_labels_by_video):
                    video_clips = attach_frame_history(video_frames, self.args.clip_length)
                    video_logits = finetuner.predict(video_clips)
                    self.test_evaluator.append_video(video_logits, video_label, video_paths, object_list)

                # if this is the user's last task, get the average performance for the user
                if (step+1) % self.args.test_tasks_per_user == 0:
                    _, current_user_stats = self.test_evaluator.get_mean_stats(current_user=True)
                    print_and_log(self.logfile, f'{self.args.test_set} user {task_dict["user_id"]} ({self.test_evaluator.current_user+1}/{self.test_queue.num_users}) stats: {stats_to_str(current_user_stats)}')
                    if (step+1) < num_test_tasks:
                        self.test_evaluator.next_user()

        # get average performance over all users
        stats_per_user, stats_per_video = self.test_evaluator.get_mean_stats()
        stats_per_user_str, stats_per_video_str = stats_to_str(stats_per_user), stats_to_str(stats_per_video)
        mean_ops_stats = self.ops_counter.get_mean_stats()
        path = path if path else self.checkpoint_dir
        print_and_log(self.logfile, f'{self.args.test_set} [{path}]\n per-user stats: {stats_per_user_str}\n per-video stats: {stats_per_video_str}\n model stats: {mean_ops_stats}\n')
        self.test_evaluator.save()
        self.test_evaluator.reset()
    
    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_stats': self.validation_evaluator.get_current_best_stats()
        }, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.validation_evaluator.replace(checkpoint['best_stats'])

if __name__ == "__main__":
    main()
