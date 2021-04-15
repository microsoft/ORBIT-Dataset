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
import sys
import time
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from utils.args import parse_args
from utils.losses import cross_entropy
from utils.ops_counter import OpsCounter
from data.dataloaders import DataLoader
from models import CNAPSRecogniser as CNAPS
from models.normalisation_layers import TaskNormI
from utils.eval_metrics import TrainEvaluator, ValidationEvaluator, TestEvaluator
from utils.logging import print_and_log, get_log_files, stats_to_str

SEED=1991
random.seed(SEED)
torch.manual_seed(SEED)
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
 
    learner = MetaLearner()
    learner.run()

class MetaLearner:
    def __init__(self):
        self.args = parse_args(mode='meta-learner')

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.model_path)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir) 

        device_id='cpu'
        self.map_location='cpu'
        if torch.cuda.is_available() and self.args.gpu>=0:
            cudnn.enabled = True
            cudnn.benchmark = False
            cudnn.deterministic = True
            device_id = 'cuda:' + str(self.args.gpu)
            torch.cuda.manual_seed_all(SEED)
            self.map_location=lambda storage, loc: storage.cuda()
        
        self.device = torch.device(device_id)
        self.ops_counter = OpsCounter()
        self.init_dataset() 
        self.init_model()
        self.init_optimizer()
        self.init_evaluators()
        self.loss = cross_entropy
    
    def init_dataset(self):
        
        dataset_info = {
            'mode': self.args.mode,
            'data_path': self.args.data_path,
            'object_cap': self.args.object_cap,
            'train_way_method' : self.args.train_way_method,
            'test_way_method' : self.args.test_way_method,
            'train_shot_methods' : [self.args.train_context_shot_method, self.args.train_target_shot_method],
            'test_shot_methods' : [self.args.test_context_shot_method, self.args.test_target_shot_method],
            'train_tasks_per_user': self.args.train_tasks_per_user,
            'test_tasks_per_user': self.args.test_tasks_per_user,
            'train_task_type' : self.args.train_task_type,
            'test_set': self.args.test_set,
            'shots' : [self.args.context_shot, self.args.target_shot],
            'video_types' : [self.args.context_video_type, self.args.target_video_type],
            'clip_length': self.args.clip_length,
            'train_num_clips': [self.args.train_context_num_clips, self.args.train_target_num_clips],
            'test_num_clips': [self.args.test_context_num_clips, self.args.test_target_num_clips],
            'subsample_factor': self.args.subsample_factor,
        }
        
        dataloader = DataLoader(dataset_info)
        self.train_queue = dataloader.get_train_queue()
        self.validation_queue = dataloader.get_validation_queue()
        self.test_queue = dataloader.get_test_queue()

    def init_model(self):
        self.model = CNAPS(self.args).to(self.device)
        self.register_extra_parameters(self.model)
        
        self.model.train()  # set encoder is always in train mode to process context data
        self.model.feature_extractor.eval()  # feature extractor is always in eval mode
        
        if self.args.use_two_gpus:
            self.model._distribute_model()
    
    def init_evaluators(self):
        self.train_metrics = ['frame_acc']
        self.evaluation_metrics = ['frame_acc', 'frames_to_recognition', 'video_acc'] 
        self.train_evaluator = TrainEvaluator(self.train_metrics)
        self.validation_evaluator = ValidationEvaluator(self.evaluation_metrics)
        self.test_evaluator = TestEvaluator(self.evaluation_metrics) 
    
    def init_optimizer(self): 
       	feature_extractor_params = list(map(id, self.model.feature_extractor.parameters()))
        base_params = filter(lambda p: id(p) not in feature_extractor_params, self.model.parameters())
        feature_extractor_lr = self.args.learning_rate*0.1 if self.args.pretrained_extractor_path else self.args.learning_rate
        self.optimizer = torch.optim.Adam([
                        {'params': base_params },
                        {'params': self.model.feature_extractor.parameters(), 'lr': feature_extractor_lr}
                        ], lr=self.args.learning_rate)
        self.optimizer.zero_grad()

    def run(self):
        if self.args.mode == 'train' or self.args.mode == 'train_test':
            for epoch in range(self.args.epochs):
                losses = []
                since = time.time()
                torch.set_grad_enabled(True)
                
                train_tasks = self.train_queue.get_tasks()
                total_steps = len(train_tasks)
                for step, task_dict in enumerate(train_tasks):
                    task_loss = self.train_task(task_dict)
                    losses.append(task_loss.detach())
                    
                    if self.args.print_by_step:
                        current_stats_str = stats_to_str(self.train_evaluator.get_current_stats())
                        print_and_log(self.logfile, 'epoch [{}/{}][{}/{}], train loss: {:.7f}, {:}'.format(epoch+1, self.args.epochs, step+1, total_steps, task_loss.item(), current_stats_str))

                    if ((step + 1) % self.args.tasks_per_batch == 0) or (step == (total_steps - 1)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        del task_loss
                
                mean_stats = self.train_evaluator.get_mean_stats()
                seconds = time.time() - since
                # print
                print_and_log(self.logfile, '-'*150)
                print_and_log(self.logfile,'epoch [{}/{}] train loss: {:.7f} {:} time/epoch: {:d}m{:02d}s' \
                                            .format(epoch + 1, self.args.epochs, \
                                            torch.Tensor(losses).mean().item(), \
                                            stats_to_str(mean_stats), \
                                            int(seconds / 60), int(seconds % 60)))
                print_and_log(self.logfile, '-'*150)
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
        context_set, context_labels, target_set, target_labels = self.unpack_task(task_dict)

        self.model.personalise(context_set, context_labels)
        target_logits = self.model(target_set)
        self.model.class_representations.clear() # reset task's params
        
        task_loss = self.loss(target_logits, target_labels) / self.args.tasks_per_batch
        if self.args.adapt_features:
            if self.args.use_two_gpus:
                regularization_term = (self.model.feature_adapter.regularization_term()).cuda(0)
            else:
                regularization_term = (self.model.feature_adapter.regularization_term())
            regularizer_scaling = 0.001
            task_loss += regularizer_scaling * regularization_term
        
        self.train_evaluator.update_stats(target_logits, target_labels)
        task_loss.backward(retain_graph=False)

        return task_loss

    def validate(self):
        
        with torch.no_grad():
            for step, task_dict in enumerate(self.validation_queue.get_tasks()):
                context_set, context_labels, target_set_by_video, target_labels_by_video = self.unpack_task(task_dict, test_mode=True)
                
                self.model.personalise(context_set, context_labels)

                for target_set, target_labels in zip(target_set_by_video, target_labels_by_video): # loop through videos
                    target_set, target_labels = self.send_to_device(target_set, target_labels)
                    target_logits = self.model(target_set, test_mode=True)
                    self.validation_evaluator.append(target_logits, target_labels)
                    del target_logits
                
                self.model.class_representations.clear() # reset task's params

                if (step+1) % self.args.test_tasks_per_user == 0:
                    self.validation_evaluator.next_user()
                    
            stats_per_user, stats_per_video = self.validation_evaluator.get_mean_stats()
            stats_per_user_str, stats_per_video_str = stats_to_str(stats_per_user), stats_to_str(stats_per_video)

            print_and_log(self.logfile, 'validation\n per-user stats: {0:}\n per-video stats: {1:}\n'.format(stats_per_user_str, stats_per_video_str))
            # save the model if validation is the best so far
            if self.validation_evaluator.is_better(stats_per_video):
                self.validation_evaluator.replace(stats_per_video)
                torch.save(self.model.state_dict(), self.checkpoint_path_validation)
                print_and_log(self.logfile, 'best validation model was updated.\n')
            
            self.validation_evaluator.reset()

    def test(self, path):

        self.init_model()
        self.model.load_state_dict(torch.load(path, map_location=self.map_location)) 
        self.ops_counter.set_base_params(self.model)

        with torch.no_grad():
            for step, task_dict in enumerate(self.test_queue.get_tasks()):
                context_set, context_labels, target_set_by_video, target_labels_by_video = self.unpack_task(task_dict, test_mode=True)

                self.model.personalise(context_set, context_labels, ops_counter=self.ops_counter)

                for target_set, target_labels in zip(target_set_by_video, target_labels_by_video): # loop through videos
                    target_set, target_labels = self.send_to_device(target_set, target_labels)
                    target_logits = self.model(target_set, test_mode=True)
                    self.test_evaluator.append(target_logits, target_labels)
                    del target_logits

                self.model.class_representations.clear() # reset task's params

                if (step+1) % self.args.test_tasks_per_user == 0:
                    self.test_evaluator.next_user()
                    
            stats_per_user, stats_per_video = self.test_evaluator.get_mean_stats()
            stats_per_user_str, stats_per_video_str = stats_to_str(stats_per_user), stats_to_str(stats_per_video)
            mean_ops_stats = self.ops_counter.get_mean_stats()
            print_and_log(self.logfile, 'test [{0:}]\n per-user stats: {1:}\n per-video stats: {2:}\n model stats: {3:}\n'.format(path, stats_per_user_str, stats_per_video_str,  mean_ops_stats))
            self.test_evaluator.save(path)
            self.test_evaluator.reset()

    def send_to_device(self, data, labels):
        data = ( data[0].to(self.device), data[1], data[2] )
        labels = labels.to(self.device)
        return data, labels     

    def unpack_task(self, task_dict, test_mode=False):
        task_dict = { k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k,v in task_dict.items() }
        context_frames, context_labels, context_paths, context_video_ids = task_dict['context_frames'], task_dict['context_labels'], task_dict['context_framepaths'], task_dict['context_video_ids']
        target_frames, target_labels, target_paths, target_video_ids = task_dict['target_frames'], task_dict['target_labels'], task_dict['target_framepaths'], task_dict['target_video_ids']

        context_frames = context_frames.to(self.device)
        context_video_ids = context_video_ids.to(self.device)
        context_paths = np.array(context_paths).reshape(-1, self.args.clip_length)
        context_set = (context_frames, context_paths, context_video_ids)
        context_labels = context_labels.to(self.device)
        
        target_paths = np.array(target_paths).reshape(-1, self.args.clip_length)

        if test_mode: # test_mode; group target set by videos
            target_set_by_video, target_labels_by_video = [], []
            unique_video_ids = torch.unique(target_video_ids)
            for video_id in unique_video_ids:
                idxs = target_video_ids == video_id
                video_clips = target_frames[idxs]
                video_paths = target_paths[idxs]
                video_labels = target_labels[idxs]
                video_ids = target_video_ids[idxs]
                target_set_by_video.append( (video_clips, video_paths, video_ids) )
                target_labels_by_video.append( video_labels )
            return context_set, context_labels, target_set_by_video, target_labels_by_video
        else:
            target_frames = target_frames.to(self.device)
            target_video_ids = target_video_ids.to(self.device)
            target_set = (target_frames, target_paths, target_video_ids)
            target_labels = target_labels.to(self.device)

            return context_set, context_labels, target_set, target_labels
     
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
 
    def register_extra_parameters(self, model):
        for module in model.modules():
            if isinstance(module, TaskNormI):
                module.register_extra_weights()


if __name__ == "__main__":
    main()
