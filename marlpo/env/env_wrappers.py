from typing import Any, Dict, Optional, Tuple, Union
import copy
from collections import defaultdict
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
from ray.tune.registry import register_env

from marlpo.env.env_utils import metadrive_to_terminated_truncated_step_api
from marlpo.utils import colorize, printPanel




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

WINDOW_SIZE = 150
NUM_NEIGHBOURS = 4



class CCEnv:
    """
    This class maintains a distance map of all agents and appends the
    neighbours' names and distances into info at each step.
    We should subclass this class to a base environment class.
    """
    
    @classmethod
    def default_config(cls):
        config = super(CCEnv, cls).default_config()
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
        super(CCEnv, self).__init__(*args, **kwargs)
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


    def _add_neighbour_info(self, agents, infos):
        for agent in agents:
            infos[agent]["agent_id"] = agent
            infos[agent]["all_agents"] = list(agents)
            neighbours, nei_distances = self._find_in_range(agent, 
                                            self.config["neighbours_distance"])
            infos[agent]["neighbours"] = neighbours
            infos[agent]["neighbours_distance"] = nei_distances

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


    def process_infos(self, agents, infos):
        self._update_distance_map()
        self._add_neighbour_info(agents, infos)
        self._last_info.update(infos)


    def proccess_infos_obs(self, infos, old_obs):
        ''' called every time original env returns obs
        Args:
            i: all agents' infos that will be added nei-infos
            obs: all agents' obs that will be proccessed to new_obs
        '''
        agents = old_obs.keys()

        # add neighbour infos
        self.process_infos(agents, infos)

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
        obs = super(CCEnv, self).reset(force_seed=force_seed)

        infos = defaultdict(dict)
        obs = self.proccess_infos_obs(infos, obs)
        
        return obs, infos


    def step(self, actions):
        obs, r, d, i = super(CCEnv, self).step(actions)
        obs = self.proccess_infos_obs(i, obs)
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


def get_ccenv(env_class):
    name = env_class.__name__

    class TMP(CCEnv, env_class):
        pass

    TMP.__name__ = name
    TMP.__qualname__ = name
    return TMP


# TODO: not functional yet
class LCFEnv(CCEnv):
    @classmethod
    def default_config(cls):
        config = super(LCFEnv, cls).default_config()
        config.update(
            dict(
                # Overwrite the CCEnv's neighbours_distance=10 to 40.
                neighbours_distance=40,

                # Two mode to compute utility for each vehicle:
                # "linear": util = r_me * lcf + r_other * (1 - lcf), lcf in [0, 1]
                # "angle": util = r_me * cos(lcf) + r_other * sin(lcf), lcf in [0, pi/2]
                # "angle" seems to be more stable!
                lcf_mode="angle",
                lcf_dist="normal",  # "uniform" or "normal"
                lcf_normal_std=0.1,  # The initial STD of normal distribution, might change by calling functions.

                # If this is set to False, then the return reward is natively the LCF-weighted coordinated reward!
                # This will be helpful in ablation study!
                return_native_reward=True,

                # Whether to force set the lcf
                force_lcf=-100,
                enable_copo=True
            )
        )
        return config

    def __init__(self, *args, **kwargs):
        super(LCFEnv, self).__init__(*args, **kwargs)
        self.lcf_map = {}
        assert hasattr(super(LCFEnv, self), "_update_distance_map")
        assert self.config["lcf_mode"] in ["linear", "angle"]
        assert self.config["lcf_dist"] in ["uniform", "normal"]
        assert self.config["lcf_normal_std"] > 0.0
        self.force_lcf = self.config["force_lcf"]

        # Only used in normal LCF distribution
        # LCF is always in range [0, 1], but the real LCF degree is in [-pi/2, pi/2].
        self.current_lcf_mean = 0.0  # Set to 0 degree.
        self.current_lcf_std = self.config["lcf_normal_std"]

        self._last_obs = None
        self._traffic_light_counter = 0

    @property
    def enable_copo(self):
        return self.config["enable_copo"]

    def get_single_observation(self, vehicle_config):
        original_obs = super(LCFEnv, self).get_single_observation(vehicle_config)

        if not self.enable_copo:
            return original_obs

        original_obs_cls = original_obs.__class__
        original_obs_name = original_obs_cls.__name__
        comm_method = self.config["communication"][COMM_METHOD]

        single_comm_dim = self.config["communication"]["comm_size"]
        if self.config["communication"]["add_pos_in_comm"]:
            single_comm_dim += 3
        comm_obs_size = single_comm_dim * self.config["communication"]["comm_neighbours"]

        add_traffic_light = self.config["add_traffic_light"]

        class LCFObs(original_obs_cls):
            @property
            def observation_space(self):
                space = super(LCFObs, self).observation_space
                assert isinstance(space, Box)
                assert len(space.shape) == 1
                length = space.shape[0] + 1

                if comm_method != "none":
                    length += comm_obs_size

                if add_traffic_light:
                    length += 1 + 2  # Global position should be put

                # Note that original metadrive obs space is [0, 1]
                space = Box(
                    low=np.array([-1.0] * length), high=np.array([1.0] * length), shape=(length, ), dtype=space.dtype
                )
                space._shape = space.shape
                return space

        LCFObs.__name__ = original_obs_name
        LCFObs.__qualname__ = original_obs_name

        # TODO: This part is not beautiful! Refactor in future release!
        # from metadrive.envs.marl_envs.tinyinter import CommunicationObservation
        # if original_obs_cls == CommunicationObservation:
        #     return LCFObs(vehicle_config, self)
        # else:
        return LCFObs(vehicle_config)

    @property
    def _traffic_light_msg(self):
        fix_interval = self.config["traffic_light_interval"]
        increment = (self._traffic_light_counter % fix_interval) / fix_interval * 0.1
        if ((self._traffic_light_counter // fix_interval) % 2) == 1:
            return 0 + increment
        else:
            return 1 - increment

    def get_agent_traffic_light_msg(self, pos):
        b_box = self.engine.current_map.road_network.get_bounding_box()
        pos0 = (pos[0] - b_box[0]) / (b_box[1] - b_box[0])
        pos1 = (pos[1] - b_box[2]) / (b_box[3] - b_box[2])
        # print("Msg: {}, Pos0: {}, Pos1 {}".format(self._traffic_light_msg, pos0, pos1))
        return np.clip(np.array([self._traffic_light_msg, pos0, pos1]), 0, 1).astype(np.float32)

    def _get_reset_return(self):
        self.lcf_map.clear()
        self._update_distance_map()
        obses = super(LCFEnv, self)._get_reset_return()

        if self.config["add_traffic_light"]:
            self._traffic_light_counter = 0
            new_obses = {}
            for agent_name, v in self.vehicles_including_just_terminated.items():
                if agent_name not in obses:
                    continue
                new_obses[agent_name] = np.concatenate(
                    [obses[agent_name], self.get_agent_traffic_light_msg(v.position)]
                )
            obses = new_obses

        ret = {}
        for k, o in obses.items():
            lcf, ret[k] = self._add_lcf(o)
            self.lcf_map[k] = lcf

        yet_another_new_obs = {}
        if self.config["communication"][COMM_METHOD] != "none":
            for k, old_obs in ret.items():
                yet_another_new_obs[k] = np.concatenate(
                    [old_obs, np.zeros((self._comm_dim * self.config["communication"]["comm_neighbours"], ))], axis=-1
                ).astype(np.float32)

            ret = yet_another_new_obs

        self._last_obs = ret
        return ret

    def step(self, actions):
        # step the environment
        o, r, d, i = super(LCFEnv, self).step(actions)
        assert set(i.keys()) == set(o.keys())
        new_obs = {}
        new_rewards = {}
        global_reward = sum(r.values()) / len(r.values())

        if self.config["add_traffic_light"]:
            self._traffic_light_counter += 1

        for agent_name, agent_info in i.items():
            assert "neighbours" in agent_info
            # Note: agent_info["neighbours"] records the neighbours within radius neighbours_distance.
            nei_rewards = [r[nei_name] for nei_name in agent_info["neighbours"]]
            if nei_rewards:
                i[agent_name]["nei_rewards"] = sum(nei_rewards) / len(nei_rewards)
            else:
                i[agent_name]["nei_rewards"] = 0.0  # Do not provide neighbour rewards if no neighbour
            i[agent_name]["global_rewards"] = global_reward

            if self.config["add_traffic_light"]:
                o[agent_name] = np.concatenate(
                    [
                        o[agent_name],
                        self.get_agent_traffic_light_msg(self.vehicles_including_just_terminated[agent_name].position)
                    ]
                )

            # add LCF into observation, also update LCF map and info.
            agent_lcf, new_obs[agent_name] = self._add_lcf(
                agent_obs=o[agent_name], lcf=self.lcf_map[agent_name] if agent_name in self.lcf_map else None
            )
            if agent_name not in self.lcf_map:
                # The agent LCF is set for the whole episode
                self.lcf_map[agent_name] = agent_lcf
            i[agent_name]["lcf"] = agent_lcf
            i[agent_name]["lcf_deg"] = agent_lcf * 90

            # lcf_map stores values in [-1, 1]
            if self.config["lcf_mode"] == "linear":
                assert 0.0 <= agent_lcf <= 1.0
                new_r = agent_lcf * r[agent_name] + (1 - agent_lcf) * agent_info["nei_rewards"]
            elif self.config["lcf_mode"] == "angle":
                assert -1.0 <= agent_lcf <= 1.0
                lcf_rad = agent_lcf * np.pi / 2
                new_r = cos(lcf_rad) * r[agent_name] + sin(lcf_rad) * agent_info["nei_rewards"]
            else:
                raise ValueError("Unknown LCF mode: {}".format(self.config["lcf_mode"]))
            i[agent_name]["coordinated_rewards"] = new_r
            i[agent_name]["native_rewards"] = r[agent_name]
            if self.config["return_native_reward"]:
                new_rewards[agent_name] = r[agent_name]
            else:
                new_rewards[agent_name] = new_r

        yet_another_new_obs = {}
        if self.config["communication"][COMM_METHOD] != "none":
            for k, old_obs in new_obs.items():
                comm_obs = i[k][COMM_CURRENT_OBS]
                if len(comm_obs) < self.config["communication"]["comm_neighbours"]:
                    comm_obs.extend(
                        [np.zeros((self._comm_dim, ))] *
                        (self.config["communication"]["comm_neighbours"] - len(comm_obs))
                    )
                yet_another_new_obs[k] = np.concatenate([old_obs] + comm_obs).astype(np.float32)

            new_obs = yet_another_new_obs

            for kkk in i.keys():
                neighbours = i[kkk]["neighbours"]
                i[kkk]["nei_obs"] = []
                for nei_index in range(self.config["communication"]["comm_neighbours"]):
                    if nei_index >= len(neighbours):
                        n = None
                    else:
                        n = neighbours[nei_index]
                    if n is not None and n in self._last_obs:
                        i[kkk]["nei_obs"].append(self._last_obs[n])
                    else:
                        i[kkk]["nei_obs"].append(None)
                i[kkk]["nei_obs"].append(None)  # Adding extra None to make sure np.array fails!

        self._last_obs = new_obs
        return new_obs, new_rewards, d, i

    def _add_lcf(self, agent_obs, lcf=None):

        if not self.enable_copo:
            return 0.0, agent_obs

        if self.force_lcf != -100:
            # Set LCF to given value
            if self.config["lcf_dist"] == "normal":
                assert -1.0 <= self.force_lcf <= 1.0
                lcf = get_np_random().normal(loc=self.force_lcf, scale=self.current_lcf_std)
                lcf = clip(lcf, -1, 1)
            else:
                lcf = self.force_lcf
        elif lcf is not None:
            pass
        else:
            # Sample LCF value from current distribution
            if self.config["lcf_dist"] == "normal":
                assert -1.0 <= self.current_lcf_mean <= 1.0
                lcf = get_np_random().normal(loc=self.current_lcf_mean, scale=self.current_lcf_std)
                lcf = clip(lcf, -1, 1)
            else:
                lcf = get_np_random().uniform(-1, 1)
        assert -1.0 <= lcf <= 1.0
        output_lcf = (lcf + 1) / 2  # scale to [0, 1]
        return lcf, np.float32(np.concatenate([agent_obs, [output_lcf]]))

    def set_lcf_dist(self, mean, std):
        assert self.enable_copo
        assert self.config["lcf_dist"] == "normal"
        self.current_lcf_mean = mean
        self.current_lcf_std = std
        assert std > 0.0
        assert -1.0 <= self.current_lcf_mean <= 1.0

    def set_force_lcf(self, v):
        assert self.enable_copo
        self.force_lcf = v


def get_lcf_env(env_class):
    name = env_class.__name__

    class TMP(LCFEnv, env_class):
        pass

    TMP.__name__ = name
    TMP.__qualname__ = name
    return TMP




def get_rllib_compatible_gymnasium_api_env(env_class, return_class=False):
    """
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
    

def get_ccppo_env(env_class, return_class=False):
    return get_rllib_compatible_gymnasium_api_env(get_ccenv(env_class), return_class=return_class)


def get_rllib_cc_env(env_class, return_class=False):
    return get_rllib_compatible_gymnasium_api_env(get_ccenv(env_class), return_class=return_class)


# TODO: refactor
def interpre_obs(obs, color: str = None):
    
    EGO_STATE_NAMES = [
        'd_left_yellow',
        'd_right_side_walk',
        'diff_heading&lane',
        'v_x',
        'steering',
        'acc_pre',
        'steering_pre',
        'yaw_rate',
        'lateral_pos',
    ]
    NAVI_INFO_NAMES = [
        'd_navi_heading',
        'd_navi_side',
        'r_checkpoint_lane',
        'clockwise',
        'angle'
    ]
    
    res = {}
    i = 0
    _dict = {}
    for name in EGO_STATE_NAMES:
        _dict[name] = obs[i]
        i += 1
    res['EGO_STATE'] = _dict

    res['-'] = '-'

    for j in range(2):
        _dict = {}
        for name in NAVI_INFO_NAMES:
            _dict[name] = obs[i]
            i += 1
        res['NAVI_'+str(j)] = _dict
        res['-'+'-'*(j+1)] = '-'
    
    res['LIDAR'] = obs[i: ]

    return res


def sns_rgb_to_rich_hex_str(color: tuple):
    palette = sns.color_palette('colorblind')
    assert color in palette
    i = palette.index(color)
    hex_color = palette.as_hex()[i]
    return hex_color

if __name__ == "__main__":
    from marlpo.utils.debug import print, printPanel
    from metadrive.component.vehicle.base_vehicle import BaseVehicle
    from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv
    from metadrive.policy.idm_policy import ManualControllableIDMPolicy
    import seaborn as sns
    import math

    config = dict(
        use_render=False,
        num_agents=2,
        manual_control=False,
        # crash_done=True,
        agent_policy=ManualControllableIDMPolicy,
        return_single_space=True,
        vehicle_config=dict(
            lidar=dict(num_lasers=72, distance=40, num_others=0),
        ),
        # == neighbour config ==
        use_dict_obs=True,
        add_compact_state=True, # add BOTH ego- & nei- compact-state simultaneously
        add_nei_state=False,
        num_neighbours=4,
        # neighbours_distance=40,
    )

    RANDOM_ACTION = False

    # cc_metadrive_env = get_ccenv(MultiAgentRoundaboutEnv)
    # cc_metadrive_new_api_env_str, cc_metadrive_new_api_env_cls = get_rllib_compatible_new_gymnasium_api_env(cc_metadrive_env, return_class=True)
    # env = cc_metadrive_new_api_env_cls(config)

    '''cc env'''
    # env_name, env_cls = get_ccppo_env(MultiAgentRoundaboutEnv, return_class=True)
    env_name, env_cls = get_rllib_cc_env(MultiAgentIntersectionEnv, return_class=True)
    # print(env_name)

    # env_cls.get_obs_shape(config)

    # exit()

    env = env_cls(config)
    o, i = env.reset()

    # print(env.config['window_size'])

    if env.current_track_vehicle:
        env.current_track_vehicle.expert_takeover = True 

    print(env.observation_space)
    print(type(env.observation_space))


    # print(o['agent0'])
    # print(i['agent0'])
    # print(o['agent1'])
    # print(i['agent1'])

    # exit()
    # env.render(mode="top_down", file_size=(800,800))

    '''lcf env'''
    # env_cls = get_lcf_env(MultiAgentIntersectionEnv)


    # print(dir(env))
    min_heading = 0
    max_heading = 0
    for i in range(2000):

        if RANDOM_ACTION:
            actions = {}
            for v in env.vehicles:
                actions[v] = env.action_space.sample()
            o, r, tm, tc, info = env.step(actions)
        else:
            obs, rew, tm, tc, infos = env.step({agent_id: [0, 0] for agent_id in env.vehicles.keys()})

        env.render(mode="top_down", file_size=(800,800))
        for agent, o in obs.items():
            msg = {}
            if isinstance(o, dict):
                msg[COMPACT_EGO_STATE] = o[COMPACT_EGO_STATE]
                msg['-'] = '-'
                msg[COMPACT_NEI_STATE] = o[COMPACT_NEI_STATE]
                # msg[EGO_STATE] = o[EGO_STATE]
                msg['*'] = '*'
                
            # named_obs = interpre_obs(o[k])
            info = infos[agent]
            vehicle: BaseVehicle = env.vehicles_including_just_terminated[agent]
            if vehicle:
                v = vehicle.velocity
                color = vehicle.panda_color
                info['agent_id'] = colorize(info['agent_id'], sns_rgb_to_rich_hex_str(color))
                # msg['OBS'] = named_obs
                # msg['-'] = '-'
                msg['INFO'] = info
                msg['--'] = '-'
                heading = int(vehicle.heading_theta / math.pi * 180)
                msg['heading'] = heading
                printPanel(msg, title='OBS')
                min_heading = min(min_heading, heading)
                max_heading = max(max_heading, heading)
        # input() 

        if tm['__all__']:
            # print(f'terminated at total {i} steps')
            print('env.episode_lengths:', env.episode_lengths)
            print('env.episode_step:', env.episode_step)
            env.reset()
            # break
        
    print(max_heading, min_heading)

    env.close()