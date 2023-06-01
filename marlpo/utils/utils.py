import sys

import rich
from rich import print, inspect
from rich.panel import Panel
# FLAG = False
# if not FLAG:
rich.get_console().width -= 30
#     FLAG = True

import json
import numbers

import numpy as np
import yaml


class SafeFallbackEncoder(json.JSONEncoder):
    def __init__(self, nan_str="null", **kwargs):
        super(SafeFallbackEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def default(self, value):
        try:
            if np.isnan(value):
                return self.nan_str

            if (type(value).__module__ == np.__name__ and isinstance(value, np.ndarray)):
                return value.tolist()

            if issubclass(type(value), numbers.Integral):
                return int(value)
            if issubclass(type(value), numbers.Number):
                return float(value)

            return super(SafeFallbackEncoder, self).default(value)

        except Exception:
            return str(value)  # give up, just stringify it (ok for logs)


def pretty_print(result):
    result = result.copy()
    result.update(config=None)  # drop config from pretty print
    # result.update(hist_stats=None)  # drop hist_stats from pretty print
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v

    cleaned = json.dumps(out, cls=SafeFallbackEncoder)
    return yaml.safe_dump(json.loads(cleaned), default_flow_style=False)

def dict_to_str(a_dict: dict):
    need_prefix = False
    prefix = ''
    # res = []
    res_str = ''
    for k, v in a_dict.items():
        if not need_prefix:
            need_prefix = True
        else:
            prefix = '\n'
        res_str += prefix + k + ': ' f'{v}'

    return res_str



def log(title: str = None, obj=None, docs=False):
    inspect(obj=obj, title=title, docs=docs)

def printPanel(msg, title=None, **kwargs):
    if isinstance(msg, str):
        pass
    elif isinstance(msg, dict):
        msg = dict_to_str(msg)
    else:
        print(f'[Warning] utils.py:printPanel -- msg with type {type(msg)}')

    print(Panel.fit(msg, title=title, **kwargs))


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
        return 6
    elif sys.platform.startswith('linux'):
        return 8
    else:
        return 1