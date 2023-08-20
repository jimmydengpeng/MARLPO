import logging
from typing import Union, Dict
import rich
from rich import print, inspect
from rich.panel import Panel

import json
import numbers

import numpy as np
import yaml

from ray.rllib.utils import try_import_torch
torch, _ = try_import_torch()

WINDOWN_WIDTH_REDUCED = False

def reduce_window_width(file_name: str = None):
    '''
    Args:
        file_name: the file's name who called this function
    '''
    global WINDOWN_WIDTH_REDUCED
    if not WINDOWN_WIDTH_REDUCED:
        file_name = file_name or __file__
        rich.get_console().width -= 30
        file_name = file_name or __file__
        WINDOWN_WIDTH_REDUCED = True
        print(f'[red]({file_name}) rich console -= 30[/]: [blue]{rich.get_console().width}[/]')

# reduce_window_width(WINDOWN_WIDTH_REDUCED)


def get_logger():
    logger = logging.getLogger("marlpo")
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s\t%(levelname)s (marlpo) %(filename)s:%(lineno)s -- %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


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


class ValueWrapper:
    GRID = 'GRID' # for table, use '-', under key & value each
    SEPARATOR = 'SEPARATOR' # for continuous separator, only one line
    DATA = 'DATA'
    GRID_DASH = '-'
    SEPARATOR_DASH = '─'

    def __init__(self, data, raw_tensor):
        self.raw_data = data 
        self.type = self.DATA
        self.str_data = self._convert(data, raw_tensor)
        

    # wrap the converting logic
    def _convert(self, data, raw_tensor) -> str:

        def contains_only(data, c):
            if not isinstance(data, str):
                return False
            return all(char == c for char in data)
        s = ''
        if self._is_tensor_type(data):
            if not raw_tensor:
                s = self._refine_tensor_type(data)
            else:
                s = '\n' + str(data)
        elif contains_only(data, '-'):
            self.type = self.GRID
            s = data
        elif contains_only(data, '*'):
            self.type = self.SEPARATOR
            s = data
        else:
            s = str(data)

        return s


    def _is_tensor_type(self, data):
        if isinstance(data, torch.Tensor) or \
        isinstance(data, np.ndarray):
            return True
        else:
            return False

    def _refine_tensor_type(self, data):
        if np.product(data.shape) < 20:
            return str(data)
        if isinstance(data, torch.Tensor):
            return str(data.shape)
        elif isinstance(data, np.ndarray):
            return type(data).__name__ + '.shape=' + str(data.shape)
        else:
            return str(data)

    def _get_max_len(self, data: str):
        lines = data.split('\n')
        max_len = 0
        for l in lines:
            max_len = max(max_len, len(l))
        return max_len
    
    def __str__(self) -> str:
        return self.str_data

    def __len__(self):
        return self._get_max_len(self.str_data)


def process_value(
        a_dict: dict, 
        indent=2, 
        use_json=False, 
        raw_tensor=False, 
        align=False
    ):
    new_dict = {}

    for k, v in a_dict.items():
        if isinstance(v, dict):
            new_dict[k] = process_value(v, raw_tensor)
        else:
            new_dict[k] = ValueWrapper(v, raw_tensor)

    return new_dict


def get_max_key_len(d) -> int:
    """Return a maximun key length of a dict"""
    max_len = 0
    for k in d:
        max_len = max(len(k), max_len)
    return max_len


def get_max_kv_len(d: Dict[str, ValueWrapper], indent, align):
    """ Return a maximum sum of length of key-value pair in a dict.
        Including colon ': '.
    """
    # for in-line indent, other shorter keys will be adjusted to this max_len
    max_key_len = get_max_key_len(d)
    # compute indent len for NEW line! Not in-line!
    cur_indent = max(max_key_len+2, indent) if align else indent
    max_len = 0
    for k, v in d.items():
        # for dict, it will be put in a new-line,
        # so the length for this dict will be indent + the dict's max_line
        if isinstance(v, dict):
            sub_len = get_max_kv_len(v, indent, align) + cur_indent
            max_len = max(max_len, sub_len)
        # for normal data, the key should be extended,
        # so the length will be the extended key length + value length  
        else:
            max_len = max(max_len, max_key_len+2+len(v))
    return max_len

def dict_to_str(
        a_dict: Dict[str, ValueWrapper], 
        indent=2, 
        use_json=False, 
        align=True,
    ):
    """Convert a dict to a string that translate every key-value pair and its
    sub-dict key-value pairs into separate rows. 

    If a key-value is '-': '-', it means a dash-line saparator, and it will be
    converted to a '-'*N line, the length will be inferred automatically.

    Args:
        a_dict: the input dict to transfer to a str.
        indent: number of space before each sub-row.
        use_json: whether to use json encoder.
        raw_tensor: output raw numbers for numpy and tensor if True; otherwize
            only output types of them.
        align: whether to align each key's colon.
    """


    max_key_len = get_max_key_len(a_dict)  
    cur_indent = max(max_key_len, indent) if align else indent

    # 1. hand-written conversion
    if not use_json:
        newline_prefix = False
        res_str = ''
        for k, v in a_dict.items():
            if not newline_prefix:
                prefix = ''
                newline_prefix = True
            else:
                prefix = '\n'

            # 1. dict
            if isinstance(v, dict):
                vv = dict_to_str(v, align=align)
                shift = cur_indent + 2
                l = [' '*shift + line for line in vv.split('\n')]
                v_str = '\n' + '\n'.join(l)
                k_str = k
                res_str += prefix + k_str + ': ' + v_str

            # 2. '---'
            # TODO: allow continuous dash separator
            elif v.type == ValueWrapper.GRID:
                max_line_len = get_max_kv_len(a_dict, indent=cur_indent, align=align)
                k_str = ValueWrapper.GRID_DASH*(max_key_len+1)
                v_str = ValueWrapper.GRID_DASH*(max_line_len-len(k_str)-1)
                res_str += prefix + k_str + ' ' + v_str

            # 3. '*'
            elif v.type == ValueWrapper.SEPARATOR:
                max_line_len = get_max_kv_len(a_dict, indent=cur_indent, align=align)
                _str = ValueWrapper.SEPARATOR_DASH*(max_line_len-1)
                res_str += prefix + _str

            # 4. normal in-line data
            elif v.type == ValueWrapper.DATA:
                if align: 
                    k_str = (k+': ').ljust(max_key_len+2)
                else:
                    k_str = k + ': '
                res_str += prefix + k_str + str(v)
               
            else:
                raise NotImplementedError

    # 2. use json dump
    else:
        res_str = json.dumps(a_dict, indent=indent)

    return res_str


def log(title: str = None, obj=None, docs=False):
    inspect(obj=obj, title=title, docs=docs)


def colorize(s: str, color: str):
    '''wrap a str with rich's color tag.
    Args:
        color: a hex color str.
    '''
    return '[bold][' + str(color) + ']' + str(s) + '[/]'


def printPanel(
    msg, 
    title=None, 
    raw_output=False, 
    align=True, 
    color: str=None, 
    **kwargs
):
    if isinstance(msg, str):
        pass
    elif isinstance(msg, int):
        msg = str(msg)
    elif isinstance(msg, dict):
        new_dict = process_value(msg, raw_tensor=raw_output)
        msg = dict_to_str(new_dict, align=align)
    else:
        print(f'[Warning] utils.py:printPanel -- msg with type {type(msg)}')

    if color:
        msg = f'[{color}]' + msg + '[/]'

    print(Panel.fit(msg, title=title, **kwargs))


def dict_to_panel(data, title=None):
    assert isinstance(data, dict)
    new_dict = process_value(data, raw_tensor=False)
    msg = dict_to_str(new_dict, align=True) 
    return Panel.fit(msg, title=title)


if __name__ == '__main__':
    import math
    t2 = torch.tensor(True)


    d0 = {
        'x': torch.randn(2,3,4),
        '-': '-',
        'xx': math.e,
    }

    d1 = {
        'a': np.random.random(size=(2, 3)),
        '-': '-',
        'aa': 3.14,
    }

    data = {
        '[red]name': 'John[/]',
        'age': 30,
        '*': '*',
        '[purple]data_0': 'DATA0[/]',
        'data_1': d0,
        'data_2': {
            'd': 'DDD',
            '-': '-',
            'ddd': d1
        }

    }

    # print(dict_to_str(d2))
    # print(dict_to_str(d2, use_json=True))

    printPanel(data, title='title',align=True, raw_output=False, color='blue')



'''
╭────────── title ──────────╮
│ name:   John              │
│ age:    30                │
│ ----------                │
│ data_0: DATA0             │
│ data_1:                   │
│   a: ndarray.shape=(2, 3) │
│   aa: 3.14                │
╰───────────────────────────╯

╭─────────── title ───────────╮
│ name:   John                │
│ age:    30                  │
│ --------------------------- │
│ data_0: DATA0               │
│ data_1:                     │
│   x: torch.Size([2, 3, 4])  │
│   y: 2.718281828459045      │
│ data_2:                     │
│   ddd: DDD                  │
│   ------------------------- │
│   d:                        │
│     a: ndarray.shape=(2, 3) │
│     ----------------------- │
│     aa: 3.14                │
╰─────────────────────────────╯
'''