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


'''
╭── obs ─────────────────────────────────────╮
│  [ego_state] [navi_0] [navi_1] [lidar]     │
│       9          5        5      72        │
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


class TrackingEnv_OLD:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.distance_tracker = defaultdict(dict)
        # self._last_info = defaultdict(dict)
        
    def reset(
        self, 
        *,
        force_seed: Optional[int] = None,
        options: Optional[dict] = None
    ):
        print('2-------', __class__.__name__)
        print('>>>', super(TrackingEnv, self))
        obs = super(TrackingEnv, self).reset()
        self.distance_tracker = defaultdict(dict)
        return obs

    def step(self, actions):
        obs, r, d, infos = super().step(actions)
        self.update_distance()
        self.add_distance_into_infos(infos)
        return obs, r, d, infos
    
    def update_distance(self):
        if hasattr(self, "vehicles_including_just_terminated"):
            vehicles = self.vehicles_including_just_terminated
        else:
            vehicles = self.vehicles  # Fallback to old version MetaDrive, but this is not accurate!
        for agent in vehicles:
            vehicle = vehicles[agent]
            if vehicle:
                position = vehicle.position
                lane, lane_index, on_lane = vehicle.navigation._get_current_lane(vehicle)
                if lane:
                    long, lat = lane.local_coordinates(position)
                    self.distance_tracker[agent][lane] = long
    
    def add_distance_into_infos(self, infos):
        for agent in infos:
            dis_dict = self.distance_tracker[agent]
            total_dis = 0
            for lane, d in dis_dict.items():
                total_dis += d
            infos[agent]['current_distance'] = total_dis


def get_rllib_compatible_gymnasium_api_env(env_class, return_class=False):
    """For old version of MetaDrive with old gym API (o,r,d,i = step())
    Args:
        env_class: A MetaDrive env class
    """
    env_name = env_class.__name__

    class NewAPIMAEnv(env_class, MultiAgentEnv):
        _agent_ids = ["agent{}".format(i) for i in range(100)] + ["{}".format(i) for i in range(10000)] + ["sdc"]

        def __init__(self, config: dict, render_mode: Optional[str] = None):
            super().__init__(config)
            super(MultiAgentEnv, self).__init__()
            self.render_mode = render_mode
            self.metadata = getattr(self, "metadata", {"render_modes": []})
            self.reward_range = getattr(self, "reward_range", None)
            self.spec = getattr(self, "spec", None)

        def reset(
                self, 
                *,
                seed: Optional[int] = None,
                options: Optional[dict] = None
            ):
            # o = super(NewAPIMAEnv, self).reset(seed=seed, options=options)
            raw_o = super(NewAPIMAEnv, self).reset(force_seed=seed)
            if isinstance(raw_o, tuple):
                return raw_o

            o = raw_o
            
            # add neighbour infos
            # infos = defaultdict(dict)
            # if hasattr(self, '_update_distance_map') and \
            #     hasattr(self, '_find_in_range'):
            #     self._update_distance_map()
            #     for agent in o:
            #         infos[agent]["agent_id"] = agent
            #         infos[agent]["all_agents"] = list(o.keys())
            #         neighbours, nei_distances = self._find_in_range(agent, 
            #                                         self.config["neighbours_distance"])
            #         infos[agent]["neighbours"] = neighbours
            #         infos[agent]["neighbours_distance"] = nei_distances
            return o, {}

        def step(self, action):
            o, r, d, i = super(NewAPIMAEnv, self).step(action)
            return metadrive_to_terminated_truncated_step_api((o, r, d, i))

        @property
        def observation_space(self):
            return convert_old_gym_space_to_gymnasium_space(super(NewAPIMAEnv, self).observation_space)

        @property
        def action_space(self):
            return convert_old_gym_space_to_gymnasium_space(super(NewAPIMAEnv, self).action_space)

        # def action_space_sample(self, agent_ids: list = None):
        #     """
        #     RLLib always has unnecessary stupid requirements that you have to bypass them by overwriting some
        #     useless functions.
        #     """
        #     return self.action_space.sample()

    NewAPIMAEnv.__name__ = env_name
    NewAPIMAEnv.__qualname__ = env_name
    register_env(env_name, lambda config: NewAPIMAEnv(config))

    if return_class:
        return env_name, NewAPIMAEnv

    return env_name
    


def count_neighbours(infos):
    epi_neighbours_list = defaultdict(list)
    for a, info in infos.items():
        epi_neighbours_list[a].append(len(info['neighbours']))
    

# ================= new stuff =====================


class TrackingEnv:
    """Add a MetaDrive env some new functions, e.g., tracking 
        each agent's distance so far every step.

       This class should be a subclass of a MetaDrive class.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_dist = defaultdict(float)

    def reset(
        self, 
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ):
        obs, infos = super().reset(seed=seed)
        self.distance_tracker = {}
        # self._update_distance_dict(obs.keys())
        self.add_distance_into_infos(infos)
        return obs, infos

    def step(self, actions):
        obs, r, tm, tc, infos = super().step(actions)
        # self._update_distance_dict(obs.keys())
        self.add_distance_into_infos(infos)
        return obs, r, tm, tc, infos
    
    def _update_distance_dict(self, agents):
        if hasattr(self, "vehicles_including_just_terminated"):
            vehicles = self.vehicles_including_just_terminated
        else:
            vehicles = self.vehicles  # Fallback to old version MetaDrive, but this is not accurate!
        
        for agent in agents:
            vehicle = vehicles.get(agent, None)
            if vehicle:
                position = vehicle.position
                lane, lane_index, on_lane = vehicle.navigation._get_current_lane(vehicle)
                if lane:
                    long, lat = lane.local_coordinates(position)
                    if agent not in self.distance_tracker:
                        self.distance_tracker[agent] = OrderedDict(
                            init_d=long
                        )
                        print('1------- long:', long)
                    self.distance_tracker[agent][lane] = long

    def add_distance_into_infos(self, infos):
        for agent in infos:
            last_dist = self.last_dist[agent]
            step_dist = 0
            if agent in self.vehicles:
                vehicle = self.vehicles[agent]
                if vehicle.lane in vehicle.navigation.current_ref_lanes:
                    current_lane = vehicle.lane
                    positive_road = 1
                else:
                    current_lane = vehicle.navigation.current_ref_lanes[0]
                    current_road = vehicle.navigation.current_road
                    positive_road = 1 if not current_road.is_negative_road() else -1
                long_last, _ = current_lane.local_coordinates(vehicle.last_position)
                long_now, lateral_now = current_lane.local_coordinates(vehicle.position)
                step_dist = (long_now - long_last) * positive_road
            cur_dist = step_dist + last_dist
            self.last_dist[agent] = cur_dist
            infos[agent]['episode_distance'] = cur_dist


def get_tracking_md_env(env_class):
    """Augment a MetaDrive env with distance tracking.
    Args:
        env_class: A MetaDrive env class.
    Returns:
        A augmented MetaDrive env class. 
    """
    name = env_class.__name__

    class TMP(TrackingEnv, env_class):
        pass

    TMP.__name__ = name
    TMP.__qualname__ = name
    return TMP


class NeighbourEnv(TrackingEnv):
    """
    This class maintains a distance map of all agents and appends the
    neighbours' names and distances into info at each step.
    We should subclass this class to a base environment class.

    Its supper class will be a MetaDrive env, e.g., 
    from MultiAgentIntersectionEnv to BaseEnv (MetaDrive).
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config["neighbours_distance"] = 40
        # ====== for backward compatible =====
        config['add_nei_state'] = False
        config['use_dict_obs'] = False
        config['add_compact_state'] = False
        return config

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))
    
    def reset(
        self, 
        *,
        seed: Union[None, int] = None,
        options: Optional[dict] = None
    ): 
        obs, infos = super().reset(seed=seed)
        self.process_infos(infos, rewards=None)
        return obs, infos

    def step(self, actions):
        obs, r, tm, tr, infos = super().step(actions)
        self.process_infos(infos, rewards=r)
        return obs, r, tm, tr, infos

    def process_infos(self, infos, rewards=None):
        self._update_distance_map()
        self._add_neighbour_info(infos, rewards)

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

    def _add_neighbour_info(self, infos, rewards=None):
        for agent in infos:
            infos[agent]["agent_id"] = agent
            infos[agent]["all_agents"] = list(infos.keys())
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


def get_neighbour_env(env_class):
    """Augment a MetaDrive env with new features,
        i.e., providing neighbors' info in the info dict for each agent. 
    Args:
        env_class: A MetaDrive env class.
    Returns:
        An augmented MetaDrive env class. 
    """
    name = env_class.__name__

    class TMP(NeighbourEnv, env_class):
        pass

    TMP.__name__ = name
    TMP.__qualname__ = name
    return TMP



def get_rllib_compatible_env(env_class, return_class=False):
    """to get a env compatible with **RLlib** and make some general modifications in **MetaDriveEnv**, e.g., turn off some burdensome or annoying logging configs:
    (1) make a MetaDrive env a subclass of RLlib's MuiltiAgentEnv, 
    (2) modify some annoying stuff in the latest version of MetaDrive,
    (3) register it to RLlib.

    Args:
        env_class: A MetaDrive env class, could be augmented (i.e., a subclass 
        of the original MetaDrive class.)
    """
    env_name = env_class.__name__ # e.g. 'MultiAgentIntersectionEnv'
    if env_name.startswith('MultiAgent'):
        env_name = env_name[len('MultiAgent'): -3] # e.g. "Intersection"

    class MAEnv(env_class, MultiAgentEnv): 
        # == rllib MultiAgentEnv requries: ==
        # _agent_ids = ["agent{}".format(i) for i in range(100)] + ["{}".format(i) for i in range(10000)] + ["sdc"]

        def __init__(self, config: dict, *args, **kwargs):
            env_class.__init__(self, config, *args, **kwargs)
            MultiAgentEnv.__init__(self)
            # == Gymnasium requires: ==
            self.render_mode = config.get('render_mode', None)
            self.metadata = getattr(self, "metadata", {"render_modes": []})
            self.reward_range = getattr(self, "reward_range", None)
            self.spec = getattr(self, "spec", None)
           
            # == turn off MetaDriveEnv logging ==
            if getattr(self, 'logger', None):
                self.logger.setLevel(logging.WARNING)

        def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
        ) -> Tuple[MultiAgentDict, MultiAgentDict]:
            obs_and_infos = super().reset(seed=seed)
            super(MultiAgentEnv, self).reset()
            return obs_and_infos
            
        def step(self, actions):
            o, r, tm, tc, i = super().step(actions)
            # == add '__all__' key to MetaDrive truncated ==
            tc['__all__'] = all(tc.values())
            return o, r, tm, tc, i 

        
    MAEnv.__name__ = env_name
    MAEnv.__qualname__ = env_name
    register_env(env_name, lambda config: MAEnv(config))

    if return_class:
        return env_name, MAEnv
    else:
        return env_name

