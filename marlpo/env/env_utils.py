import time
import math
from collections import namedtuple, defaultdict, OrderedDict
from typing import List, Union
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import seaborn as sns

from metadrive import (
    MetaDriveEnv,
    MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv, MultiAgentTollgateEnv,
    MultiAgentBottleneckEnv, MultiAgentParkingLotEnv
)
from metadrive.component.vehicle.base_vehicle import BaseVehicle

from utils.debug import colorize


SCENES = {
    "roundabout": MultiAgentRoundaboutEnv,
    "intersection": MultiAgentIntersectionEnv,
    "tollgate": MultiAgentTollgateEnv,
    "bottleneck": MultiAgentBottleneckEnv,
    "parkinglot": MultiAgentParkingLotEnv,
}


def get_metadrive_ma_env_cls(scene_name: str):
    return SCENES[scene_name]


# for single-agent env
def metadrive_to_terminated_truncated_step_api(step_returns):
    """Function to transform step returns to new step API irrespective of input API.

    Args:
        step_returns (tuple): Items returned by step(). Can be (obs, rew, done, info) or (obs, rew, terminated, truncated, info)
        is_vector_env (bool): Whether the step_returns are from a vector environment
    """
    if len(step_returns) == 5:
        return step_returns
    else:
        assert len(step_returns) == 4
        observations, rewards, dones, infos = step_returns

        # for single agent
        if not isinstance(dones, dict):
            return (
                observations,
                rewards,
                dones,
                False,
                infos,
            )
        # for multi-agent
        else:
            _dones = []
            truncated = {}
            for k, v in dones.items():
                _dones.append(int(v))
                truncated[k] = False
            dones['__all__'] = np.alltrue(_dones)
            return (
                observations,
                rewards,
                dones,
                truncated,
                infos,
            )

def metadrive_dict_to_OrderedDict(data: dict) -> OrderedDict:
    assert isinstance(data, dict)
    return OrderedDict(data)


def average_episode_reward(env, num_episode=100, expert_takeover=True):
    # t0 = time.time()
    # max_r = 0
    # min_r = 0
    epi_steps = []
    epi_rews = []
    epi_r = 0

    # rate = namedtuple('rates', ('succ', 'out_of_road', 'crash_vehicle'))(succ=[], out_of_road=[], crash_vehicle=[])
    rate = defaultdict(list)

    o, info = env.reset()
    env.vehicle.expert_takeover = expert_takeover

    for epi in tqdm(range(num_episode)):
        while True:
            o, r, terminated, truncated, info = env.step(env.action_space.sample())
            epi_r += r
            
            
            if terminated or truncated:
                epi_steps.append(info['episode_length'])
                rate['succ'].append(int(info['arrive_dest']))
                rate['out_of_road'].append(int(info['out_of_road']))
                rate['crash_vehicle'].append(int(info['crash_vehicle']))
                rate['max_step'].append(int(info['max_step']))
                
                env.reset()
                epi_rews.append(epi_r)
                epi_r = 0
                env.current_track_vehicle.expert_takeover = expert_takeover
                break
    
    rate_info = []
    for k, v in rate.items():
        _v = f'{round(np.count_nonzero(v)/num_episode*100, 2)}%'
        rate_info.append([k, _v])
    
    res = [
        ['total_episode', num_episode],
        ['average_reward', np.mean(epi_rews)],
        ['max', np.max(epi_rews)],
        ['min', np.min(epi_rews)],
    ] + rate_info
    print(tabulate(res, tablefmt="rounded_grid"))


def sns_rgb_to_rich_hex_str(color: tuple):
    '''
    Args:
        color: (r, g, b) | each in [0, 1]
    '''
    palette = sns.color_palette('colorblind')
    assert color in palette
    i = palette.index(color)
    hex_color = palette.as_hex()[i]
    return hex_color


def get_agent_color_in_env_for_rich(
    agent: str, 
    env: MetaDriveEnv, 
) -> str:
    vehicle: BaseVehicle = env.vehicles_including_just_terminated[agent]
    if vehicle:
        return sns_rgb_to_rich_hex_str(vehicle.panda_color)
    else:
        print(f'[Warning] failed getting agent_id <{agent}> \'s color from env!')
        return '#33ff33' # light green


def colorize_dict(
    dict_: dict, 
    color: str, # a rich compatible color
    keys: Union[str, List[str]] = None,
) -> str:
    if isinstance(keys, str): keys = [keys]
    for k in keys:
        dict_[k] = colorize(dict_[k], color)
    return dict_


def colorize_info_for_agent(
    agent: str, 
    info: dict, 
    env: MetaDriveEnv, 
    keys: Union[str, List[str]] = None,
) -> str:
    """wrap a rich's color tag around an agent's id in info dict
       with its own rendering color.
    """
    color = get_agent_color_in_env_for_rich(agent, env)
    colorize_dict(agent, info, color, keys)
    return color


# TODO: refactor
def interpreted_obs(obs: np.ndarray, config: dict, color: str = None):
    """Make the obs array human readable.

    Args:
        obs: obs from env.
        color: str | hex color for rich.
    
    Returns:
        A dict that can be print through printPanel().
    """
    def parse_multi_group_obs(
        obs, 
        start_idx: int, 
        num_groups: int, 
        group_entries: tuple, 
        group_name: str,
    ) -> dict:
        groups = {}
        i = start_idx
        for j in range(num_groups):
            tmp = {}
            for name in group_entries:
                tmp[name] = obs[i]
                i += 1
            groups[group_name+'_'+str(j)] = tmp
            if j != num_groups-1:
                groups['-'+'-'*(j+1)] = '-'
        return groups, i
    
    EGO_STATE = (
        'd_left_yellow',
        'd_right_side_walk',
        'diff_heading&lane',
        'v_x',
        'steering',
        'acc_pre',
        'steering_pre',
        'yaw_rate',
        'lateral_pos',
    )

    NAVI_INFO = (
        'd_navi_heading',
        'd_navi_side',
        'r_checkpoint_lane',
        'clockwise',
        'angle',
    )

    OTHER_VEHICLES_INFO = (
        'd_heading',
        'd_side',
        'v_heading',
        'v_side',
    )

    num_lasers = config['vehicle_config']['lidar']['num_lasers']
    num_others = config['vehicle_config']['lidar']['num_others']

    assert len(EGO_STATE) + 2*len(NAVI_INFO) + \
            num_others*len(OTHER_VEHICLES_INFO) + num_lasers \
            == len(obs)
    
    res = {}
    i = 0
    # == 1. EGO
    tmp = {}
    for name in EGO_STATE:
        tmp[name] = obs[i]
        i += 1
    res['EGO_STATE'] = tmp
    res['*'] = '*'

    res['NAVI_INFO'], i = parse_multi_group_obs(obs, i, 2, NAVI_INFO, 'NAVI')
    res['**'] = '*'

    res['OTHERS'], i = parse_multi_group_obs(obs, i, 4, OTHER_VEHICLES_INFO, 'OTHERS')
    res['***'] = '*'
    
    res['LIDAR'] = obs[i: ]

    return res
