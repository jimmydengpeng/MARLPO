from typing import Any, Dict, Optional, Tuple
import copy
from collections import defaultdict
from math import cos, sin

import gym as old_gym
import gymnasium as gym
from gymnasium.spaces import Box

import numpy as np
from metadrive.utils import get_np_random, clip


from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.gym import convert_old_gym_space_to_gymnasium_space, check_old_gym_env
from ray.tune.registry import register_env
from marlpo.env.env_utils import metadrive_to_terminated_truncated_step_api



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

EGO_DIM = 9
NAVI_DIM = 5
OLD_STATE_DIM = EGO_DIM + 2*NAVI_DIM # 9 + 5x2 = 19 
NEW_STATE_DIM = OLD_STATE_DIM + 2 # add additional x, y pos of each vehicle

WINDOW_SIZE = 150

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
                communication=dict(comm_method="none", comm_size=4, comm_neighbours=4, add_pos_in_comm=False),
                add_traffic_light=False,
                traffic_light_interval=30,
                # == neighbours config ==
                neighbour_states=False, # if True, obs returned will contain neighbours' states
                num_neighbours=4, # determine observation_space, i.e., how many neighbours can be observed
            )
        )
        return config

    def __init__(self, *args, **kwargs):
        super(CCEnv, self).__init__(*args, **kwargs)
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))
        if self.config["communication"][COMM_METHOD] != "none":
            self._comm_obs_buffer = defaultdict()

        if self.config["communication"]["add_pos_in_comm"]:
            self._comm_dim = self.config["communication"]["comm_size"] + 3
        else:
            self._comm_dim = self.config["communication"]["comm_size"]

        self._last_info = defaultdict(dict)

    def _get_reset_return(self):
        if self.config["communication"][COMM_METHOD] != "none":
            self._comm_obs_buffer = defaultdict()
        return super(CCEnv, self)._get_reset_return()

    @property
    def action_space(self):
        old_action_space = super(CCEnv, self).action_space
        if not self.config["communication"][COMM_METHOD] != "none":
            return old_action_space
        assert isinstance(old_action_space, Dict)
        new_action_space = Dict(
            {
                k: Box(
                    low=single.low[0],
                    high=single.high[0],
                    dtype=single.dtype,

                    # We are not using self._comm_dim here!
                    shape=(single.shape[0] + self.config["communication"]["comm_size"], )
                )
                for k, single in old_action_space.spaces.items()
            }
        )
        return new_action_space

    @property
    def observation_space(self) -> gym.Space:
        """
        Return observation spaces of active and controllable vehicles
        :return: Dict
        """
        old_obs_space = super(CCEnv, self).observation_space
        if not self.config['neighbour_states']:
            return old_obs_space

        if self.config["return_single_space"]:
            assert isinstance(old_obs_space, old_gym.spaces.Box)
            if self.config['neighbour_states']:
                lidar_dim = old_obs_space.shape[0] - OLD_STATE_DIM # 72 by default, maybe 0
                length = (self.config['num_neighbours']+1) * NEW_STATE_DIM + lidar_dim
                obs_space = old_gym.spaces.Box(
                    low=np.array([-1.0] * length), 
                    high=np.array([1.0] * length), 
                    shape=(length, ), 
                    dtype=old_obs_space.dtype 
                )
                    
                return obs_space
        
        # TODO:
        else:
            raise NotImplementedError


    def _insert_pos_in_obs(self, o: Dict[str, np.ndarray], infos): 
        """ 1. cache last x, y pos in info
            2. add x, y position in each vehicle's obs
            NOTE: excluding vehicles just terminated (can't get Vehicle Instance)
        """

        new_o = {}
        for agent, obs in o.items():
            if hasattr(self, "vehicles_including_just_terminated"):
                vehicles = self.vehicles_including_just_terminated
            else:
                vehicles = self.vehicles 
            if vehicles[agent]:
                pos = vehicles[agent].position
            else:
                pos = self._last_info[agent].get('last_pos', (0, 0))
                # msg = {}
                # msg['agent'] = agent
                # msg['vehicles'] = vehicles
                # msg['obs'] = o[agent]
                # msg['pos'] = pos
                # msg['info'] = infos[agent]
                # msg['last_info'] = self._last_info[agent]
                # printPanel(msg)
            infos[agent]['last_pos'] = pos
            new_o[agent] = np.insert(obs, 0, np.array(pos)/WINDOW_SIZE)

        # print(infos)
        return new_o


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
            nei_states = np.zeros((num_neighbours, NEW_STATE_DIM))
            nei_s_list = [ obs[nei][:NEW_STATE_DIM] for nei in neighbours ]
            
            for i, nei_s in zip(range(num_neighbours), nei_s_list):
                nei_states[i] = nei_s
            new_o[agent] = np.insert(o, NEW_STATE_DIM, nei_states.flatten())
        return new_o
               
    def reset(
        self, 
        *,
        force_seed: Optional[int] = None,
        options: Optional[dict] = None
    ):
        o = super(CCEnv, self).reset(force_seed=force_seed)

        # add neighbour infos
        i = defaultdict(dict)
        self._update_distance_map()
        self._add_neighbour_info(o.keys(), i)

        if self.config['neighbour_states']:
            o = self._add_neighbour_state(self._insert_pos_in_obs(o, i), i)
        self._last_info.update(i)
        return o, i


    def step(self, actions):

        if self.config["communication"][COMM_METHOD] != "none":
            comm_actions = {k: v[2:] for k, v in actions.items()}
            actions = {k: v[:2] for k, v in actions.items()}

        o, r, d, i = super(CCEnv, self).step(actions)

        self._update_distance_map(dones=d)

        self._add_neighbour_info(o.keys(), i)
        # for kkk, info in i.items():
        #     info["agent_id"] = kkk
        #     info["all_agents"] = list(i.keys())

        #     neighbours, nei_distances = self._find_in_range(kkk, self.config["neighbours_distance"])
        #     info["neighbours"] = neighbours
        #     info["neighbours_distance"] = nei_distances

        #     nei_rewards = [r[nei_name] for nei_name in neighbours]
        #     info['nei_rewards'] = nei_rewards
        #     if nei_rewards:
        #         info["nei_average_rewards"] = np.mean(nei_rewards)
        #     else:
        #         info["nei_average_rewards"] = 0.0
        #     # i[agent_name]["global_rewards"] = global_reward


        if self.config['neighbour_states']:
            o = self._add_neighbour_state(self._insert_pos_in_obs(o, i), i)

        self._last_info.update(i)
        
        return o, r, d, i

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
            env_class.__init__(self, config)
            MultiAgentEnv.__init__(self)
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


def interpre_obs(obs):
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


if __name__ == "__main__":
    from marlpo.utils.debug import print, printPanel
    from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv
    from metadrive.policy.idm_policy import ManualControllableIDMPolicy

    config = dict(
        use_render=False,
        num_agents=30,
        manual_control=False,
        # crash_done=True,
        agent_policy=ManualControllableIDMPolicy,
        return_single_space=True,
        vehicle_config=dict(
            lidar=dict(num_lasers=0, distance=40, num_others=0),
        ),
        # == neighbour config ==
        neighbour_states=True,
        num_neighbours=2,
        neighbours_distance=20,
    )

    RANDOM_ACTION = False

    # cc_metadrive_env = get_ccenv(MultiAgentRoundaboutEnv)
    # cc_metadrive_new_api_env_str, cc_metadrive_new_api_env_cls = get_rllib_compatible_new_gymnasium_api_env(cc_metadrive_env, return_class=True)
    # env = cc_metadrive_new_api_env_cls(config)

    '''cc env'''
    # env_name, env_cls = get_ccppo_env(MultiAgentRoundaboutEnv, return_class=True)
    env_name, env_cls = get_rllib_cc_env(MultiAgentIntersectionEnv, return_class=True)
    # print(env_name)
    env = env_cls(config)
    o, i = env.reset()

    # print(env.config['window_size'])

    if env.current_track_vehicle:
        env.current_track_vehicle.expert_takeover = True 

    print(env.observation_space)

    # print(o['agent0'])
    # print(i['agent0'])
    # print(o['agent1'])
    # print(i['agent1'])
    # exit()

    # env.render(mode="top_down", file_size=(800,800))

    '''lcf env'''
    # env_cls = get_lcf_env(MultiAgentIntersectionEnv)


    # print(dir(env))
    for i in range(20000):

        if RANDOM_ACTION:
            actions = {}
            for v in env.vehicles:
                actions[v] = env.action_space.sample()
            o, r, tm, tc, info = env.step(actions)
        else:
            o, r, tm, tc, info = env.step({agent_id: [0, 0] for agent_id in env.vehicles.keys()})
        env.render(mode="top_down", file_size=(800,800))
        # input() 
        for k in o:
            named_obs = interpre_obs(o[k])
            _info = info[k]
            break
        msg = {}
        msg['OBS'] = named_obs
        msg['-'] = '-'
        msg['INFO'] = _info
        # printPanel(msg, title='OBS')
        # input()
        # for agent, obs in o.items():
            # print(obs)
            # print(info[agent])
            # print('-'*10)
        # input() 
        # print('='*20)

        # for a in info:
        #     if info[a].get('arrive_dest', False):
        #         print(o[a]) 

        if tm['__all__']:
            # print(f'terminated at total {i} steps')
            print('env.episode_lengths:', env.episode_lengths)
            print('env.episode_step:', env.episode_step)
            env.reset()
            # break

    env.close()