import sys

from rich import print, inspect
import rich
rich.get_console().width -= 30


def log(title: str = None, obj=None, docs=False):
    inspect(obj=obj, title=title, docs=docs)


def get_other_training_resources():
    if sys.platform.startswith('darwin'):
        # No GPUs
        return dict(
            num_gpus=0,
        )
    elif sys.platform.startswith('linux'):
        # 1 GPU
        return dict(
            num_gpus=1,
            num_gpus_per_learner_worker=1,
        )
    else:
        return {}
