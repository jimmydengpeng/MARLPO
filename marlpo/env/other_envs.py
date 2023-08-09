from typing import Any, Dict, Optional, Tuple, Union
import copy
from collections import defaultdict, OrderedDict
import logging
from math import cos, sin, pi
import numpy as np

import gym as old_gym
import gym.spaces as old_gym_spaces
import gymnasium as gym
from gymnasium.spaces import Box

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.utils import get_np_random, clip

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.gym import convert_old_gym_space_to_gymnasium_space, check_old_gym_env
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from .env_utils import metadrive_to_terminated_truncated_step_api
from utils.debug import printPanel



COMM_ACTIONS = "comm_actions"
COMM_PREV_ACTIONS = "comm_prev_actions"

# prev_obs_{t-1} is the concatenation of neighbors' message comm_action_{t-1}
COMM_PREV_OBS = "comm_prev_obs"

# current_obs_t is the concatenation of neighbors' message comm_action_{t-1}
COMM_CURRENT_OBS = "comm_current_obs"
COMM_PREV_2_OBS = "comm_prev_2_obs"

COMM_LOGITS = "comm_logits"
COMM_LOG_PROB = "comm_log_prob"
ENV_PREV_OBS = "env_prev_obs"

COMM_METHOD = "comm_method"

NEI_OBS = "nei_obs"

# === KEYS ===
EGO_STATE = 'ego_state'
EGO_NAVI = 'ego_navi'
LIDAR = 'lidar'
COMPACT_EGO_STATE = 'compact_ego_state'
COMPACT_NEI_STATE = 'compact_nei_state'
NEI_STATE = 'nei_state'
NEI_REWARDS = "nei_rewards"


''' obs:
╭─ before ───────────────────────────────────╮
│  [ego_state] [navi_0] [navi_1] [lidar]     │
│       9          5        5      72        │
╰────────────────────────────────────────────╯

                        ╭─>[x, y, vx, vy, heading, speed, steering, yaw_rate]
                        ╭─>[presence, x, y, vx, vy, heading, steering, yaw_rate]
                        │
╭─ after ───────────────┼────────────────────╮
│  [compact_ego_state] ─╯                8   │ 
│ ────────────────────────────────────────── │ 
│  [ego-state]                           9   │ 
│  [ego-navi]                         2x 5   │ 
│ ────────────────────────────────────────── │ 
│  [compact-nei-state]...             Nx 8   │ 
│  [nei-state]...                     Nx 9   │
│ ────────────────────────────────────────── │
│  [lidar]                              72   │
╰────────────────────────────────────────────╯
'''
# unchanged:
STATE_DIM = 9
NAVI_DIM = 5
NUM_NAVI = 2

# old:
# OLD_STATE_NAVI_DIM = OLD_EGO_DIM + 2*NAVI_DIM # 9 + 5x2 = 19 
# NEW_STATE_DIM = OLD_STATE_NAVI_DIM + 2 # add additional x, y pos of each vehicle 

# new
COMPACT_STATE_DIM = 8 # x, y, vx, vy, heading, speed, steering, yaw_rate

# TODO: OBS Structure dict: {EGO: dim, EGO_NAVI_0, EGO_NAVI_1, NEI_EGO ..., LIDAR}
# can also be passed to custom model's config

WINDOW_SIZE = 151
NUM_NEIGHBOURS = 4


# Compact States Env Wrapper
class CSEnv:
    """
    This class maintains a distance map of all agents and appends the
    neighbours' names and distances into info at each step.
    We should subclass this class to a base environment class.
    """
    
    @classmethod
    def default_config(cls):
        config = super(CSEnv, cls).default_config()
        # NOTE: this config is set to 40 in LCFEnv
        config["neighbours_distance"] = 40
        config.update(
            dict(
                use_dict_obs=False,
                # == neighbours config ==
                add_compact_state=False, # if use necessary nei states
                # augment_ego_state=False,
                add_nei_state=False, # if True, obs returned will contain neighbour's state
                # nei_navi=False, # if True, will include nei's navi info
                num_neighbours=NUM_NEIGHBOURS, # up-to how many neighbours can be observed
            )
        )
        return config
    
    def validate_config(self):
        self.num_neighbours = self.config.get('num_neighbours', 0)
        self.lidar_dim = self.config['vehicle_config']['lidar']['num_lasers']
        # 如果没有增加任何观测，并且也没有使用字典观测空间，就返回原始空间
        self.return_original_obs = not (
            self.config.get('add_compact_state', False) 
            or self.config.get('add_nei_state', False)
        ) and not self.config.get('use_dict_obs', False) 

        self.obs_dim = self._compute_obs_dim()
        self.old_obs_shape = self._get_original_obs_shape()
        self.new_obs_shape, _obs_dim = self._get_new_obs_shape(self.old_obs_shape)
        assert self.obs_dim == _obs_dim

    def _compute_obs_dim(self) -> int: # called only when __init__
        ''' 根据 self.config 计算观测空间的维度
        '''
        old_obs_space: old_gym_spaces.Space = super().observation_space
        dim = np.product(old_obs_space.shape)
        if self.config.get('add_compact_state', False):
            dim += (self.num_neighbours+1) * COMPACT_STATE_DIM
        if self.config.get('add_nei_state', False):
            dim += (self.num_neighbours) * STATE_DIM
        return dim

    def _get_original_obs_shape(self):
        return {
            'ego_state': (STATE_DIM, ) ,
            'ego_navi': (NUM_NAVI, NAVI_DIM),
            'lidar': (self.lidar_dim, ),
        }

    def _get_new_obs_shape(self, old_shape) -> Tuple[Dict, int]:
        ''' get a dict of obs structure, including obs's names & dims
        '''
        res = {}
        if self.config.get('add_compact_state', False):
            res['compact_ego_state'] = (COMPACT_STATE_DIM, )
            res['ego_state'] = old_shape['ego_state']
            res['ego_navi'] = old_shape['ego_navi']
            res['compact_nei_state'] = (self.num_neighbours, COMPACT_STATE_DIM)
        else:
            res['ego_state'] = old_shape['ego_state']
            res['ego_navi'] = old_shape['ego_navi']

        if self.config.get('add_nei_state', False):
            res['nei_state'] = (self.num_neighbours, STATE_DIM)

        res['lidar'] = old_shape['lidar']

        total_dim = 0
        for name, shape in res.items():
            total_dim += np.product(shape)

        return res, total_dim


    @classmethod
    def get_obs_shape(cls, env_config):
        config = cls.default_config().update(env_config)
        res = {
            'ego_state': (STATE_DIM, ) ,
            'ego_navi': (NUM_NAVI, NAVI_DIM),
            'lidar': (config['vehicle_config']['lidar']['num_lasers'], )
        }
        if config.get('add_compact_state', False):
            res['compact_ego_state'] = (COMPACT_STATE_DIM, )
            res['compact_nei_state'] = (config.get('num_neighbours', NUM_NEIGHBOURS), COMPACT_STATE_DIM)

        if config.get('add_nei_state', False):
            res['nei_state'] = (config.get('num_neighbours', NUM_NEIGHBOURS), STATE_DIM)

        return res


    def __init__(self, *args, **kwargs):
        super(CSEnv, self).__init__(*args, **kwargs)
        self.validate_config()
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))
        self._last_info = defaultdict(dict)


    def get_dict_obs_space(
        self, 
        obs_shape: Dict[str, Tuple[int, ...]], 
        dtype,
    ) -> old_gym_spaces.Dict:
        ''' 根据输入的观测空间的形状字典，返回一个字典空间
        '''
        dict_space = {}
        for name, shape in obs_shape.items():
            dict_space[name] = old_gym_spaces.Box(
                low=-1.0, 
                high=1.0, 
                shape=shape, 
                dtype=dtype 
            )
        return old_gym_spaces.Dict(dict_space)
    

    def get_box_obs_space(self, obs_dim: int, dtype) -> old_gym_spaces.Box:
        return old_gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(obs_dim, ), 
            dtype=dtype)


    @property
    def observation_space(self):
        """
        Return observation spaces of active and controllable vehicles
        :return: Dict
        """
        old_obs_space = super().observation_space
        if self.return_original_obs:
            return old_obs_space
        assert self.config["return_single_space"]
        # 1. dict_obs_shape
        if self.config['use_dict_obs']:
            return self.get_dict_obs_space(self.new_obs_shape, old_obs_space.dtype)
        # 2. box_obs_shape
        else:
            return self.get_box_obs_space(self.obs_dim, old_obs_space.dtype)


    def _add_neighbour_info(self, agents, infos, rewards=None):
        for agent in agents:
            # agent_info = infos[agent]
            infos[agent]["agent_id"] = agent
            infos[agent]["all_agents"] = list(agents)
            neighbours, nei_distances = self._find_in_range(agent, 
                                            self.config["neighbours_distance"])
            infos[agent]["neighbours"] = neighbours
            infos[agent]["neighbours_distance"] = nei_distances

            if rewards:
                nei_rewards = [rewards[nei] for nei in neighbours]
                infos[agent][NEI_REWARDS] = nei_rewards
                # if nei_rewards:
                    # agent_info["nei_rewards"] = sum(nei_rewards) / len(nei_rewards)
                # else:
                #     i[agent_name]["nei_rewards"] = 0.0  # Do not provide neighbour rewards if no neighbour
            else:
                infos[agent][NEI_REWARDS] = []


    def _add_neighbour_state(self, obs, infos):
        new_o = {}
        for agent, o in obs.items():
            neighbours = infos[agent]["neighbours"]
            num_neighbours = self.config['num_neighbours']
            nei_state_dim = self._compute_nei_state_dim()
            nei_states = np.zeros((num_neighbours, nei_state_dim))
            all_nei_states = [ obs[nei][:nei_state_dim] for nei in neighbours ]
            
            for i, nei_s in zip(range(num_neighbours), all_nei_states):
                nei_states[i] = nei_s
            new_o[agent] = np.insert(o, nei_state_dim, nei_states.flatten())
        return new_o


    def obs_to_dict(
        self, 
        o: np.ndarray, 
        obs_shape: Dict[str, Tuple[int, ...]],
    ):
        ''' 将一个单智能体观测从numpy数组按照字典形状转换为一个字典
        '''
        res = {}
        pre_idx = 0
        for name, shape in obs_shape.items():
            cur_idx = pre_idx + np.product(np.array(shape))
            res[name] = o[pre_idx: cur_idx].reshape(shape)
            pre_idx = cur_idx
        return res


    def flatten_obs_dict(
        self, 
        obs_dict: Dict[str, Dict[str, np.ndarray]],
    ):
        res = {}
        for agent, o_dict in obs_dict.items():
            tmp = []
            for o in o_dict.values():
                tmp.append(o.flatten())
            res[agent] = np.concatenate(tmp)
        return res
    

    def get_compact_state(self, agent: str, o_dict: Dict[str, np.ndarray]):
        ''' [presence, x, y, vx, vy, heading, steering, yaw_rate] '''

        ego_state = o_dict[EGO_STATE]

        if hasattr(self, "vehicles_including_just_terminated"):
            vehicles = self.vehicles_including_just_terminated
        else:
            vehicles = self.vehicles  # Fallback to old version MetaDrive, but this is not accurate!
        vehicle: BaseVehicle = vehicles[agent]

        # x, y, v_x, v_y
        if vehicle: # could be none
            pos = vehicle.position # metadrive.utils.math_utils.Vector
            pos = np.array(pos)/WINDOW_SIZE
            if pos[0] > 1 or pos[1] > 1:
                printPanel({'pos': pos})

            velocity = vehicle.velocity / vehicle.max_speed # numpy.ndarray
            for i in range(len(velocity)):
                velocity[i] = clip(velocity[i], 0.0, 1.0)             

            heading = np.array([((vehicle.heading_theta / pi * 180 + 90) / 180)])

            self._last_info[agent]['pos'] = pos
            self._last_info[agent]['velocity_array'] = velocity
            self._last_info[agent]['heading'] = heading

            # (vehicle.speed_km_h / vehicle.max_speed_km_h)
        else:
            pos = self._last_info[agent].get('pos', np.array((0., 0.)))
            velocity = self._last_info[agent].get('velocity_array', np.array((0., 0.)))
            heading = self._last_info[agent].get('heading', np.array([0.]))
        
        # presence
        presence = np.ones((1))
        # speed, yaw_rate, steering
        speed = np.zeros((1)) + ego_state[3]
        steering = np.zeros((1)) + ego_state[4]
        yaw_rate = np.zeros((1)) + ego_state[7]

        return np.concatenate((presence, pos, velocity, heading, steering, yaw_rate))


    def _add_compact_state(
        self, 
        obs_dict: Dict[str, Dict[str, np.ndarray]],
    ):
        for agent, o_dict in obs_dict.items():
            o_dict['compact_ego_state'] = self.get_compact_state(agent, o_dict)

            compact_nei_states = np.zeros((self.num_neighbours, COMPACT_STATE_DIM))
            neighbours = self._last_info[agent].get('neighbours', [])

            for i, nei in zip(range(self.num_neighbours), neighbours):
                compact_nei_states[i] = self.get_compact_state(nei, obs_dict[nei])
            
            o_dict['compact_nei_state'] = compact_nei_states
            
    def _add_nei_state(
        self,
        obs_dict: Dict[str, Dict[str, np.ndarray]],
    ):
        for agent, o_dict in obs_dict.items():
            nei_states = np.zeros((self.num_neighbours, STATE_DIM))
            neighbours = self._last_info[agent].get('neighbours', [])

            for i, nei in zip(range(self.num_neighbours), neighbours):
                nei_states[i] = obs_dict[nei][EGO_STATE]
            o_dict['nei_state'] = nei_states


    def process_obs(
        self, 
        old_obs: Dict[str, np.ndarray],
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        ''' 根据自身的配置，处理原始观测，并加入必要的信息'''
        # 1. transform original obs to dict
        obs_dict = {}
        for agent, o in old_obs.items():
            obs_dict[agent] = self.obs_to_dict(o, self.old_obs_shape)

        # 2. add additional obs if needed
        if self.config['add_compact_state']:
            self._add_compact_state(obs_dict)
        if self.config['add_nei_state']:
            self._add_nei_state(obs_dict)

        # 3. return Dict or Box obs according to config
        # 3.1 dict obs
        if self.config['use_dict_obs']:
            return obs_dict
        # 3.2 flatten obs
        else:
            return self.flatten_obs_dict(obs_dict)


    def process_infos(self, agents, infos, rewards=None):
        self._update_distance_map()
        self._add_neighbour_info(agents, infos, rewards)
        self._last_info.update(infos)


    def proccess_infos_obs(self, infos, old_obs, rewards=None):
        ''' called every time original env returns obs
        Args:
            i: all agents' infos that will be added nei-infos
            obs: all agents' obs that will be proccessed to new_obs
        '''
        agents = old_obs.keys()

        # add neighbour infos
        self.process_infos(agents, infos, rewards)

        # add obs and/or change type to dict
        new_obs = self.process_obs(old_obs)
        old_obs = old_obs if self.return_original_obs else new_obs
        return old_obs


    def reset(
        self, 
        *,
        force_seed: Optional[int] = None,
        options: Optional[dict] = None
    ):
        obs = super(CSEnv, self).reset(force_seed=force_seed)

        infos = defaultdict(dict)
        obs = self.proccess_infos_obs(infos, obs)
        
        return obs, infos


    def step(self, actions):
        obs, r, d, i = super(CSEnv, self).step(actions)
        obs = self.proccess_infos_obs(i, obs, rewards=r)
        return obs, r, d, i


    def _find_in_range(self, v_id, distance):
        if distance <= 0:
            return [], []
        max_distance = distance
        dist_to_others = self.distance_map[v_id]
        dist_to_others_list = sorted(dist_to_others, key=lambda k: dist_to_others[k])
        ret = [
            dist_to_others_list[i] for i in range(len(dist_to_others_list))
            if dist_to_others[dist_to_others_list[i]] < max_distance
        ]
        ret2 = [
            dist_to_others[dist_to_others_list[i]] for i in range(len(dist_to_others_list))
            if dist_to_others[dist_to_others_list[i]] < max_distance
        ]
        return ret, ret2


    def _update_distance_map(self, dones=None):
        self.distance_map.clear()
        if hasattr(self, "vehicles_including_just_terminated"):
            vehicles = self.vehicles_including_just_terminated
            # if dones is not None:
            #     assert (set(dones.keys()) - set(["__all__"])) == set(vehicles.keys()), (dones, vehicles)
        else:
            vehicles = self.vehicles  # Fallback to old version MetaDrive, but this is not accurate!
        keys = [k for k, v in vehicles.items() if v is not None]
        for c1 in range(0, len(keys) - 1):
            for c2 in range(c1 + 1, len(keys)):
                k1 = keys[c1]
                k2 = keys[c2]
                p1 = vehicles[k1].position
                p2 = vehicles[k2].position
                distance = np.linalg.norm(p1 - p2)
                self.distance_map[k1][k2] = distance
                self.distance_map[k2][k1] = distance


def get_csenv(env_class):
    name = env_class.__name__

    class TMP(CSEnv, env_class):
        pass

    TMP.__name__ = name
    TMP.__qualname__ = name
    return TMP
