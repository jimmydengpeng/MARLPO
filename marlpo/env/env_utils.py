import time
import math
from collections import namedtuple, defaultdict, OrderedDict
from tqdm import tqdm
from tabulate import tabulate
import numpy as np

from metadrive import (
    MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv, MultiAgentTollgateEnv,
    MultiAgentBottleneckEnv, MultiAgentParkingLotEnv
)

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
            # if infos["arrive_dest"] or infos['max_step']:
            #     terminated = True
            # else:
            #     terminated = dones

            # if infos['crash'] or infos['crash_vehicle'] or infos['out_of_road'] or infos['']:
            #     truncated = True
            # else:
            #     truncated = False

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


if __name__ == '__main__':
    try:
        from .metadrive_env import RLlibMetaDriveEnv 
    except:
        from metadrive_env import RLlibMetaDriveEnv 

    config = dict(
        use_render=False,
        manual_control=True,
        traffic_density=0.1,
        environment_num=10,
        random_agent_model=False,
        random_lane_width=False,
        random_lane_num=False,
        map=7,
        horizon=1000,
        start_seed=np.random.randint(0, 1000),
        # ===== Reward Scheme =====
        success_reward=10.0,
        out_of_road_penalty=10.0, #5.0,
        crash_vehicle_penalty=10.0, #5.0,
        crash_object_penalty=10.0, #5.0,
        driving_reward=1.0, #1.0,
        speed_reward=0.1,
        use_lateral_reward=False,
    )
    env = RLlibMetaDriveEnv(config)
    average_episode_reward(env, num_episode=10)