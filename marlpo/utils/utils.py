import sys


def get_other_training_resources():
    if sys.platform.startswith('darwin'):
        # No GPUs
        return dict(
            num_gpus=0,
        )
    elif sys.platform.startswith('linux'):
        # 1 GPU
        return dict(
            num_gpus=0,
            # num_gpus_per_learner_worker=,
        )
    else:
        return {}

def get_num_workers():
    if sys.platform.startswith('darwin'):
        return 4
    elif sys.platform.startswith('linux'):
        return 4
    else:
        return 0