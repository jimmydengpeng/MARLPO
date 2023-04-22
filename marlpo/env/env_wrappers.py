from typing import Any, Dict, Optional, Tuple
import copy
from collections import defaultdict
from math import cos, sin

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


class CCEnv:
    """
    This class maintains a distance map of all agents and appends the
    neighbours' names and distances into info at each step.
    We should subclass this class to a base environment class.
    """
    @classmethod
    def default_config(cls):
        config = super(CCEnv, cls).default_config()
        # Note that this config is set to 40 in LCFEnv
        config["neighbours_distance"] = 40

        config.update(
            dict(
                communication=dict(comm_method="none", comm_size=4, comm_neighbours=4, add_pos_in_comm=False),
                add_traffic_light=False,
                traffic_light_interval=30,
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

    def step(self, actions):

        if self.config["communication"][COMM_METHOD] != "none":
            comm_actions = {k: v[2:] for k, v in actions.items()}
            actions = {k: v[:2] for k, v in actions.items()}

        o, r, d, i = super(CCEnv, self).step(actions)
        self._update_distance_map(dones=d)
        for kkk in i.keys():
            i[kkk]["all_agents"] = list(i.keys())

            neighbours, nei_distances = self._find_in_range(kkk, self.config["neighbours_distance"])
            i[kkk]["neighbours"] = neighbours
            i[kkk]["neighbours_distance"] = nei_distances

            if self.config["communication"][COMM_METHOD] != "none":
                i[kkk][COMM_CURRENT_OBS] = []
                for n in neighbours[:self.config["communication"]["comm_neighbours"]]:
                    if n in comm_actions:
                        if self.config["communication"]["add_pos_in_comm"]:
                            ego_vehicle = self.vehicles_including_just_terminated[kkk]
                            nei_vehicle = self.vehicles_including_just_terminated[n]
                            relative_position = ego_vehicle.projection(nei_vehicle.position - ego_vehicle.position)
                            dis = np.linalg.norm(relative_position)
                            extra_comm_obs = [
                                dis / 20, ((relative_position[0] / dis) + 1) / 2, ((relative_position[1] / dis) + 1) / 2
                            ]
                            tmp_comm_obs = np.concatenate([comm_actions[n], np.clip(np.asarray(extra_comm_obs), 0, 1)])
                        else:
                            tmp_comm_obs = comm_actions[n]
                        i[kkk][COMM_CURRENT_OBS].append(tmp_comm_obs)
                    else:
                        i[kkk][COMM_CURRENT_OBS].append(np.zeros((self._comm_dim, )))

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




def get_rllib_compatible_new_gymnasium_api_env(env_class, return_class=False):
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
            o = super(NewAPIMAEnv, self).reset(force_seed=seed)
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


if __name__ == "__main__":
    from rich import print
    from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv
    from marlpo.algo_ccppo import get_ccppo_env

    config = dict(
        use_render=False,
        num_agents=10,
        # "manual_control": True,
        # crash_done=True,
        # "agent_policy": ManualControllableIDMPolicy
        return_single_space=True,
    )

    # cc_metadrive_env = get_ccenv(MultiAgentRoundaboutEnv)
    # cc_metadrive_new_api_env_str, cc_metadrive_new_api_env_cls = get_rllib_compatible_new_gymnasium_api_env(cc_metadrive_env, return_class=True)
    # env = cc_metadrive_new_api_env_cls(config)
    env = get_ccppo_env(MultiAgentRoundaboutEnv)
    print(env.reset())
    # exit()
    for i in range(2000):
        actions = {}
        for v in env.vehicles:
            actions[v] = env.action_space.sample()

        # o, r, ter, i = env.step(actions)
        o, r, ter, trunc, i = env.step(actions)
        env.render(mode="top_down", file_size=(800,800))
        print(i)
        # input()
        if ter['__all__']:
            print(i)
            print(env.episode_lengths)
            print(env.episode_step)
            break

    env.close()