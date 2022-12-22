# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import argparse

FRAME_ANNOTATION_OPTIONS = ["object_not_present_issue", "framing_issue", "viewpoint_issue", "blur_issue", "occlusion_issue", "overexposed_issue", "underexposed_issue"]
NEGATED_FRAME_ANNOTATION_OPTIONS = [f"no_{ann}" for ann in FRAME_ANNOTATION_OPTIONS]
BOUNDING_BOX_OPTIONS = ["object_bounding_box"]
ALL_FRAME_ANNOTATION_OPTIONS = FRAME_ANNOTATION_OPTIONS + NEGATED_FRAME_ANNOTATION_OPTIONS + ["no_issues"] + ["mixed_issues"]

def parse_args(learner='default'):

    parser = argparse.ArgumentParser()

    # default parameters
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="Directory to save checkpoints to.")
    parser.add_argument("--data_path", required=True, help="Path to ORBIT root directory.")
    parser.add_argument("--test_set", default='test', choices=['validation', 'test'],
                        help="Test set to sample test tasks.")

    # model parameters
    parser.add_argument("--model_path", "-m", default=None,
                        help="Path to model to load and test.")
    parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                        help="Whether to run training only, testing only, or both training and testing.")
    parser.add_argument("--feature_extractor", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "efficientnet_v2_s", "vit_s_32", "vit_b_32", "vit_b_32_clip"],
                        help="Feature extractor backbone (default: efficientnet_b0).")
    parser.add_argument("--learn_extractor", action="store_true",
                        help="If True, learns all parameters of feature extractor.")
    parser.add_argument("--adapt_features", action="store_true",
                        help="If True, learns FiLM layers for feature adaptation.")
    parser.add_argument("--classifier", default="proto", choices=["linear", "versa", "proto", "proto_cosine", "mahalanobis"],
                        help="Classifier head to use (default: proto).")
    parser.add_argument("--logit_scale", type=float, default=1.0,
                        help="Scale factor for logits (default: 1.0).")

    # data parameters
    parser.add_argument("--train_way_method", type=str, default="random", choices=["random", "max"],
                        help="Method to sample classes for a train task (default: random).")
    parser.add_argument("--test_way_method", type=str, default="max", choices=["random", "max"],
                        help="Method to sample classes for a test/validation task (default: max).")
    parser.add_argument("--train_object_cap", type=int, default=15,
                        help="Cap on objects sampled per train task (default: 15).")
    parser.add_argument("--test_object_cap", type=int, default=15,
                        help="Cap on objects sampled per test task (default: 15).")
    parser.add_argument("--train_context_shot_method", type=str, default="random", choices=["specific", "fixed", "random", "max"],
                        help="Method to sample context shots for a train task (default: random).")
    parser.add_argument("--train_target_shot_method", type=str, default="random", choices=["specific", "fixed", "random", "max"],
                        help="Method to sample target shots for a train task (default: random).")
    parser.add_argument("--test_context_shot_method", type=str, default="max", choices=["specific", "fixed", "random", "max"],
                        help="Method to sample context shots for a test/validation task (default: max).")
    parser.add_argument("--test_target_shot_method", type=str, default="max", choices=["specific", "fixed", "random", "max"],
                        help="Method to sample target shots for a test/validation task (default: max).")
    parser.add_argument("--context_shot", type=int, default=5,
                        help="If train/test_context_shot_method = specific/fixed, number of videos per object for context set (default: 5).")
    parser.add_argument("--target_shot", type=int, default=2,
                        help="If train/test_context_shot_method = specific/fixed, number of videos per object for target set (default: 2).")
    parser.add_argument("--with_train_shot_caps", action="store_true",
                        help="Caps videos per objects sampled in train tasks.")
    parser.add_argument("--context_video_type", type=str, default='clean', choices=['clean'],
                        help="Video type for context set (default: clean).")
    parser.add_argument("--target_video_type", type=str, default='clutter', choices=['clutter', 'clean'],
                        help="Video type for target set (default: clutter).")
    parser.add_argument("--subsample_factor", type=int, default=30,
                        help="Factor to subsample video by if sampling clip uniformly (default: 30).")
    parser.add_argument("--train_context_clip_method", type=str, default='uniform', choices=['random', 'random_200', 'max', 'uniform'],
                        help="Method to sample clips per context video for a train task (default: uniform).")
    parser.add_argument("--train_target_clip_method", type=str, default='random', choices=['random', 'random_200', 'max'],
                        help="Method to sample clips per target video for a train task (default: random).")
    parser.add_argument("--test_context_clip_method", type=str, default='random', choices=['random', 'random_200', 'max', 'uniform'],
                        help="Method to sample clips per context video for a test/validation task (default: random).")
    parser.add_argument("--test_target_clip_method", type=str, default='random_200', choices=['random', 'random_200', 'max'],
                        help="Method to sample clips per target video for a test/validation task (default: random_200).")
    parser.add_argument("--clip_length", type=int, default=1,
                        help="Number of frames to sample per clip (default: 1).")
    parser.add_argument("--frame_size", type=int, default=224, choices=[224],
                        help="Frame size (default: 224).")
    parser.add_argument("--annotations_to_load", nargs='+', type=str, default=[], choices=FRAME_ANNOTATION_OPTIONS+BOUNDING_BOX_OPTIONS,
                        help="Annotations to load per frame (default: None).")
    parser.add_argument("--train_filter_context", nargs='+', type=str, default=[], choices=ALL_FRAME_ANNOTATION_OPTIONS,
                        help="Criteria to filter context frames in train tasks by (default: []).")
    parser.add_argument("--train_filter_target", nargs='+', type=str, default=[], choices=ALL_FRAME_ANNOTATION_OPTIONS,
                        help="Criteria to filter target frames in validation/test tasks by (default: []).")
    parser.add_argument("--test_filter_context", nargs='+', type=str, default=[], choices=ALL_FRAME_ANNOTATION_OPTIONS,
                        help="Criteria to filter context frames in validation/test tasks by (default: []).")
    parser.add_argument("--test_filter_target", nargs='+', type=str, default=[], choices=ALL_FRAME_ANNOTATION_OPTIONS,
                        help="Criteria to filter target frames in train tasks by (default: []).")
    parser.add_argument("--train_task_type", type=str, default="user_centric", choices=["user_centric", "object_centric"],
                        help="Sample train tasks as user-centric or object-centric (default: user_centric).")
    parser.add_argument("--num_train_tasks", type=int, default=50,
                        help="Number of train tasks per user (if train_task_type = user_centric) or per object (if train_task_type = object_centric) per epoch (default: 50).")
    parser.add_argument("--num_val_tasks", type=int, default=15,
                        help="Number of validation tasks per user (default: 15).")
    parser.add_argument("--num_test_tasks", type=int, default=50,
                        help="Number of test tasks per user (default: 50).")

    # training parameters
    parser.add_argument("--seed", type=int, default=1991,
                        help="Random seed (default: 1991).")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size when processing context and target set. Used for training/testing FineTuner and CNAPs/ProtoNets with LITE (default: 256).")
    parser.add_argument("--tasks_per_batch", type=int, default=16,
                        help="Number of tasks between parameter optimization.")
    parser.add_argument("--with_lite", action="store_true",
                        help="If True, trains with LITE.")
    parser.add_argument("--num_lite_samples", type=int, default=16,
                        help="Number of context clips per task to back-propagate with LITE training (default: 16)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu id to use (default: 0, cpu: <0)")
    parser.add_argument("--print_by_step", action="store_true",
                        help="Print training by step (otherwise print by epoch).")

    # optimization parameters
    parser.add_argument("--epochs", "-e", type=int, default=15,
                        help="Number of training epochs (default: 15).")
    parser.add_argument("--validation_on_epoch", type=int, default=1,
                        help="Epoch to turn on validation (default: 1).")
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-6,
                        help="Learning rate (default: 5e-6).")
    parser.add_argument("--extractor_lr_scale", type=float, default=1.0,
                        help="Factor to scale learning rate for feature extractor (default: 1.0).")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                        help="Optimization method to use (default: adam).")
    parser.add_argument("--weight_decay", type=float, default=0.2,
                        help="Weight decay value for optimizer (default: 0.2).")

    adam_group = parser.add_argument_group("Adam optimizer hyperparameters")
    adam_group.add_argument("--epsilon", type=float, default=1e-6,
                        help="Epsilon value for Adam optimizer (default: 1e-6).")
    adam_group.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.98),
                        help="Beta values for Adam optimizer (default: (0.9, 0.98).")

    sgd_group = parser.add_argument_group("SGD optimizer hyperparameters")
    sgd_group.add_argument("--momentum", type=float, default=0.0,
                        help="Momentum value for SGD optimizer (default: 0.0).")

    parser.add_argument("--scheduler", dest="sched", type=str, default="multistep", choices=["step", "multistep", "cosine"],
                        help="Type of learning rate scheduler (default: multistep.")
    parser.add_argument("--warmup_lr", type=float, default=1e-6,
                        help="Warmup learning rate for scheduler (default: 1e-6).")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs for scheduler (default: 5).")

    step_group = parser.add_argument_group("Step/multi-step optimizer hyperparameters")
    step_group.add_argument("--decay_epochs", type=int, default=15,
                        help="Epoch period LR is decayed for step scheduler (default: 15).")
    step_group.add_argument("--decay_rate", type=float, default=0.5,
                        help="Learning rate decay factor for step scheduler (default: 0.5).")
    
    cosine_group = parser.add_argument_group("Cosine optimizer hyperparameters")
    cosine_group.add_argument("--cooldown_epochs", type=int, default=0,
                        help="Number of cooldown epochs for cosine scheduler (default: 0).")
    cosine_group.add_argument("--lr_k_decay", type=float, default=0.1,
                        help="Learning rate decay value for cosine scheduler (default: 0.1).")
    cosine_group.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate for cosine scheduler (default: 1e-6).")


    # specific parameters
    if learner == 'multi-step-learner':
        finetune_group = parser.add_argument_group("Finetuning hyperparameters to use for personalization")
        finetune_group.add_argument("--personalize_num_grad_steps", type=int, default=50,
                        help="Number of gradient steps for personalization (default: 50).")
        finetune_group.add_argument("--personalize_learning_rate", type=float, default=0.007,
                        help="Learning rate for personalization (default: 0.007).")
        finetune_group.add_argument("--personalize_optimizer", type=str, choices=["sgd", "adam"], default="adam",
                        help="Optimizer type for personalization (default: adam).")
        finetune_group.add_argument("--personalize_weight_decay", type=float, default=0.0,
                        help="Weight decay for personalization (default: 0.0).")
        finetune_group.add_argument("--personalize_extractor_lr_scale", type=float, default=1.0,
                        help="Factor to scale learning rate for feature extractor during personalization (default: 1.0).")
        finetune_group.add_argument("--personalize_epsilon", type=float, default=1e-8,
                        help="Epsilon value for Adam optimizer during personalization (default 1e-8).")
        finetune_group.add_argument("--personalize_betas", type=float, nargs=2, default=(0.9, 0.999),
                        help="Beta values for Adam optimizer during personalization (default: (0.9, 0.999).")
        finetune_group.add_argument("--personalize_momentum", type=float, default=0.0,
                        help="Momentum for SGD optimizer during personalization (default: 0.0).")

    args = parser.parse_args()
    args.train_filter_context = expand_issues(args.train_filter_context)
    args.train_filter_target = expand_issues(args.train_filter_target)
    args.test_filter_context = expand_issues(args.test_filter_context)
    args.test_filter_target = expand_issues(args.test_filter_target)
    if args.feature_extractor == 'efficientnet_b0':
        args.frame_norm_method = 'imagenet'
    elif args.feature_extractor in ['efficientnet_v2_s', 'vit_s_32', 'vit_b_32']:
        args.frame_norm_method = 'imagenet_inception'
    elif args.feature_extractor == 'vit_b_32_clip':
        args.frame_norm_method = 'openai_clip'
    verify_args(learner, args)
    return args

def expand_issues(original_arg):
    if "no_issues" in original_arg:
        return NEGATED_FRAME_ANNOTATION_OPTIONS
    if "mixed_issues" in original_arg:
        return FRAME_ANNOTATION_OPTIONS
    return original_arg

    return new_arg

def verify_args(learner, args):
    # define some print out colours
    cred = "\33[31m"
    cyellow = "\33[33m"
    cend = "\33[0m"

    if 'train' in args.mode and not args.learn_extractor and not args.adapt_features:
        sys.exit('{:}error: at least one of "--learn_extractor" and "--adapt_features" must be used during training{:}'.format(cred, cend))

    if learner == 'multi-step-learner':
        if 'train' in args.mode:
            sys.exit('{:}error: Only "--mode test" is supported for multi-step-learner.py{:}'.format(cred, cend))

        if args.with_lite:
            print('{:}warning: "--with_lite" is not relevant for multi-step-learner.py. Normal batching is used instead{:}'.format(cyellow, cend))
