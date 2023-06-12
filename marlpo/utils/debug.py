from typing import Union
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

from ray.rllib.utils import try_import_torch
torch, _ = try_import_torch()

def is_tensor_type(data):
    if isinstance(data, torch.Tensor) or \
       isinstance(data, np.ndarray):
        return True
    else:
        return False

def refine_tensor_type(data):
    if isinstance(data, torch.Tensor):
        if len(data.shape) == 0:
            return str(data)
        elif len(data.shape) == 1 and data.shape[0] == 1:
            return str(data)
        else:
            return data.shape
    elif isinstance(data, np.ndarray):
        return type(data).__name__ + '.shape=' + str(data.shape)

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

def dict_to_str(a_dict: dict, indent=2, use_json=False):
    # 1. hand-written conversion
    if not use_json:
        need_prefix = False
        res_str = ''
        for k, v in a_dict.items():
            if not need_prefix:
                prefix = ''
                need_prefix = True
            else:
                prefix = '\n'

            if isinstance(v, dict):
                vv = dict_to_str(v)
                # v = '\n' + v
                v = []
                for l in vv.split('\n'):
                    v.append(' '*indent + l)
                v = '\n' + '\n'.join(v)
            if is_tensor_type(v):
                v = refine_tensor_type(v)
                
            res_str += prefix + k + ': ' f'{v}'

    # 2. use json dump
    else:
        res_str = json.dumps(a_dict, indent=indent)

    return res_str



def log(title: str = None, obj=None, docs=False):
    inspect(obj=obj, title=title, docs=docs)


def printPanel(msg, title=None, **kwargs):
    if isinstance(msg, str):
        pass
    elif isinstance(msg, int):
        msg = str(msg)
    elif isinstance(msg, dict):
        msg = dict_to_str(msg)
    else:
        print(f'[Warning] utils.py:printPanel -- msg with type {type(msg)}')

    print(Panel.fit(msg, title=title, **kwargs))


if __name__ == '__main__':
    import math

    array = np.random.random(size=(2, 3))
    t = torch.randn(2,3,4)
    t2 = torch.tensor(True)

    data = {
        'name': 'John',
        'age': 30,
        'city': t2
    }

    d1 = dict(
        x=t,
        y=math.e,
        z=data,
    )
    d2 = dict(
        a=array,
        b='hello',
        c=3.14,
        d=d1
    )
    print(dict_to_str(d2))
    # print(dict_to_str(d2, use_json=True))