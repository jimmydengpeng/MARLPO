import copy
import os
import pickle

import numpy as np

import ray
from ray import air, tune
from ray.tune import CLIReporter

ray.init()

def train(
    trainer,
    config,
    stop,
    exp_name,
    num_seeds=1,
    num_gpus=0,
    test_mode=False,
    suffix="",
    checkpoint_freq=10,
    keep_checkpoints_num=None,
    start_seed=0,
    local_mode=False,
    save_pkl=True,
    custom_callback=None,
    max_failures=1,
    # wandb support is removed!
    wandb_key_file=None,
    wandb_project=None,
    wandb_team="copo",
    wandb_log_config=True,
    init_kws=None,
    **kwargs
):

    if (keep_checkpoints_num is not None) and (not test_mode) and (keep_checkpoints_num != 0):
        assert isinstance(keep_checkpoints_num, int)
        checkpoint_config = air.CheckpointConfig(
            num_to_keep=keep_checkpoints_num,
            checkpoint_frequency=checkpoint_freq,
            checkpoint_score_attribute="RewardMean", # see callbacks.py
        )
    else:
        checkpoint_config = air.CheckpointConfig(checkpoint_at_end=True)

    verbose = 3 if test_mode else 1

    tuner = tune.Tuner(
        trainer,
        run_config=air.RunConfig(
            name=exp_name,
            local_dir="./exp_results",
            stop=stop,
            checkpoint_config=checkpoint_config,
            verbose=verbose,
        ),
        param_space=config,
    )



    results = tuner.fit()

