# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F
from timm.scheduler import create_scheduler

def cross_entropy(test_logits, test_labels, reduction='mean'):
    return F.cross_entropy(test_logits, test_labels, reduction=reduction)
    
def init_optimizer(model, lr, optimizer_type, args={}, extractor_lr_scale=0.1):
    feature_extractor_params = list(map(id, model.feature_extractor.parameters()))
    base_params = filter(lambda p: id(p) not in feature_extractor_params, model.parameters())
    if optimizer_type == 'adam':
        optimizer_fn = torch.optim.Adam
        extra_args = dict(
                        eps=getattr(args, 'epsilon', 1e-08),
                        weight_decay=getattr(args, 'weight_decay', 0.0),
                        betas=getattr(args, 'betas', (0.9, 0.999))
                        )
    elif optimizer_type == 'sgd':
        optimizer_fn = torch.optim.SGD
        extra_args = dict(
                        momentum=getattr(args, 'momentum', 0.0),
                        weight_decay=getattr(args, 'weight_decay', 0.0),
                        )
    optimizer = optimizer_fn([
                        {'params': base_params},
                        {'params': model.feature_extractor.parameters(), 'lr_scale': extractor_lr_scale}
                        ], lr=lr, **extra_args)
    optimizer.zero_grad()
    return optimizer

def init_scheduler(optimizer, args):
    if args.sched == 'multistep':
        args.decay_milestones = list(range(0, args.epochs, args.decay_epochs))
    if args.sched == 'cosine':
        args.warmup_prefix = True
    scheduler, _ = create_scheduler(args, optimizer)
    return scheduler

def get_curr_learning_rates(optimizer):
    lrs = []
    for group in optimizer.param_groups:
        lr = group['lr']
        lrs.append(lr)
    return lrs
    
