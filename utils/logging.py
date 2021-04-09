# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from datetime import datetime

def print_and_log(log_file, message):
    print(message)
    log_file.write(message + '\n')

def get_log_files(checkpoint_dir, model_path):
    """
    Function that takes a path to a checkpoint directory and returns a reference to a logfile and paths to the
    fully trained model and the model with the best validation score.
    """
    if model_path:
        model_dir = os.path.split(model_path)[0]
        verify_checkpoint_dir(model_dir)
    
    checkpoint_dir = os.path.join(checkpoint_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(checkpoint_dir)

    checkpoint_path_best = os.path.join(checkpoint_dir, 'best.pt')
    checkpoint_path_final = os.path.join(checkpoint_dir, 'final.pt')
    logfile_path = os.path.join(checkpoint_dir, 'log.txt')
    if os.path.isfile(logfile_path):
        logfile = open(logfile_path, "a", buffering=1)
    else:
        logfile = open(logfile_path, "w", buffering=1)

    return checkpoint_dir, logfile, checkpoint_path_best, checkpoint_path_final

def verify_checkpoint_dir(checkpoint_dir):
    # verify that the checkpoint directory and file exists
    if not os.path.exists(checkpoint_dir):
        print("Can't resume/test for checkpoint. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
        sys.exit()

def stats_to_str(stats):
    s=''
    for stat, scores in stats.items():
        if isinstance(scores, list):
            s+='{0:}: {1:.2f} ({2:.2f}) '.format(stat, scores[0]*100, scores[1]*100)
        else:
            s+='{0:}: {1:.2f} '.format(stat, scores*100)
    return s
