# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import argparse

def parse_args(mode='default'):

    parser = argparse.ArgumentParser()
   
    # default parameters
    parser.add_argument("--checkpoint_dir", default='./checkpoint', help="Directory to save checkpoint to.")
    parser.add_argument("--data_path", required=True, help="Path to ORBIT root directory.")
    parser.add_argument("--feature_extractor", type=str, default="resnet_18", choices=["resnet_18"],
                        help="Feature extractor backbone (default: resnet_18).")
    parser.add_argument("--pretrained_extractor_path", default='./features/pretrained/resnet_18_imagenet_84.pth', 
                        help="Path to pretrained feature extractor model.")
    parser.add_argument("--adapt_features", action="store_true",
                        help="If True, learns FiLM layers for feature adaptation.")
    parser.add_argument("--learn_extractor", action="store_true",
                        help="If True, learns all parameters of feature extractor at 0.1 of learning rate.")
    parser.add_argument("--batch_normalisation", choices=["basic",  "task_norm-i"], default="basic", 
                        help="Normalisation layer to use (default: basic).")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0001,
                        help="Learning rate (default: 0.0001).")
    parser.add_argument("--tasks_per_batch", type=int, default=16,
                        help="number of tasks between parameter optimization.")
    parser.add_argument("--subsample_factor", type=int, default=1,
                        help="Factor to subsample video by before sampling clips (default: 1).")
    parser.add_argument("--clip_length", type=int, default=8,
                        help="Number of frames to sample per clip (default: 8).")
    parser.add_argument("--train_context_num_clips", type=int, default=4,
                        help="Number of clips to sample per context video for a train task (default: 4).")
    parser.add_argument("--train_target_num_clips", type=int, default=4,
                        help="Number of clips to sample per target video for a train task (default: 4).")
    parser.add_argument("--test_context_num_clips", type=int, default=8,
                        help="Number of clips to sample per context video for a test/validation task (default: 8).")
    parser.add_argument("--test_target_num_clips", type=str, default='max',
                        help="Sample all overlapping clips per target video for a test/validation task (default: max).")
    parser.add_argument("--train_way_method", type=str, default="random", choices=["random", "max"],
                        help="Method to sample classes for a train task (default: random).")
    parser.add_argument("--test_way_method", type=str, default="max", choices=["random", "max"],
                        help="Method to sample classes for a test/validation task (default: max).")
    parser.add_argument("--train_context_shot_method", type=str, default="random", choices=["specific", "fixed", "random", "max"],
                        help="Method to sample context shots for a train task (default: random).")
    parser.add_argument("--train_target_shot_method", type=str, default="random", choices=["specific", "fixed", "random", "max"],
                        help="Method to sample target shots for a train task (default: random).")
    parser.add_argument("--test_context_shot_method", type=str, default="max", choices=["specific", "fixed", "random", "max"],
                        help="Method to sample context shots for a test/validation task (default: max).")
    parser.add_argument("--test_target_shot_method", type=str, default="max", choices=["specific", "fixed", "random", "max"],
                        help="Method to sample target shots for a test/validation task (default: max).")
    parser.add_argument("--train_task_type", type=str, default="user_centric", choices=["user_centric", "object_centric"],
                        help="Sample train tasks as user-centric or object-centric.")
    parser.add_argument("--train_tasks_per_user", type=int, default=50,
                        help="Number of train tasks per user (default: 50).")
    parser.add_argument("--test_tasks_per_user", type=int, default=5,
                        help="Number of test tasks per user (default: 5).")
    parser.add_argument("--test_set", default='test', choices=['validation', 'test'], 
                        help="Test set to sample test tasks.")
    parser.add_argument("--context_shot", type=int, default=5,
                        help="If train/test_context_shot_method = specific/fixed, number of videos per object for context set (default: 5).")
    parser.add_argument("--target_shot", type=int, default=2,
                        help="If train/test_context_shot_method = specific/fixed, number of videos per object for target set (default: 2).")
    parser.add_argument("--context_video_type", type=str, default='clean', choices=['clean'],
                        help="Video type for context set (default: clean).")
    parser.add_argument("--target_video_type", type=str, default='clutter', choices=['clutter', 'clean'],
                        help="Video type for target set (default: clutter).")
    parser.add_argument("--object_cap", type=int, default=10,
                        help="Cap on objects sampled per train task (default: 10).")
    parser.add_argument("--gpu", type=int, default=0, 
                        help="gpu id to use (default: 0, cpu: <0)")
    parser.add_argument("--print_by_step", action="store_true",
                        help="Print training by step (otherwise print by epoch).")
    parser.add_argument("--use_two_gpus", dest="use_two_gpus", default=False, action="store_true",
                        help="If True, do model parallelism over 2 GPUs.")
    parser.add_argument("--model_path", "-m", default=None,
                        help="Path to model to load and resume/test.")
    parser.add_argument("--epochs", "-e", type=int, default=10, 
                        help="Number of training epochs (default: 10).")
    parser.add_argument("--validation_on_epoch", type=int, default=5,
                        help="Epoch to turn on validation (default: 5).")
    parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                        help="Whether to run training only, testing only, or both training and testing.")

    if 'finetune-learner' in mode:
        parser.add_argument("--num_grad_steps", type=int, default=50,
                        help="Number of finetuning steps to take (default: 50).")
        parser.add_argument("--inner_learning_rate", "--inner_lr", type=float, default=0.1,
                        help="Learning rate for finetuning (default: 0.1).")
    elif 'meta-learner' in mode:
        parser.add_argument("--classifier", default="versa", choices=["versa", "proto"],
                        help="Classifier head to use (default: versa).")
        if 'gradient' in mode:
            parser.add_argument("--num_grad_steps", type=int, default=15,
                        help="Number of gradient steps for inner loop (default: 15).")
            parser.add_argument("--inner_learning_rate", "--inner_lr", type=float, default=0.1,
                        help="Learning rate for inner loop (default: 0.1).")
             
    args = parser.parse_args()
    verify_args(mode, args)
    return args

def verify_args(mode, args): 
    if 'gradient' in mode or 'finetune' in mode:
        if 'train' in args.mode and not args.learn_extractor and not args.adapt_features:
            sys.exit('error: at least one of "--learn_extractor" and "--adapt_features" must be used')
