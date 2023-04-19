from typing import Any, Dict, Optional, Tuple

from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray.rllib.utils.gym import convert_old_gym_space_to_gymnasium_space, check_old_gym_env

from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv
)

try:
    from env_utils import metadrive_to_terminated_truncated_step_api, metadrive_dict_to_OrderedDict
except:
    from .env_utils import metadrive_to_terminated_truncated_step_api, metadrive_dict_to_OrderedDict


class RLlibMultiAgentMetaDrive(MultiAgentEnv):
    envs = dict(
        roundabout=MultiAgentRoundaboutEnv,
        intersection=MultiAgentIntersectionEnv,
        tollgate=MultiAgentTollgateEnv,
        bottleneck=MultiAgentBottleneckEnv,
        parkinglot=MultiAgentParkingLotEnv,
        pgma=MultiAgentMetaDrive
    )

    def __init__(self, config: dict, render_mode: Optional[str] = None):
        super().__init__()
        if 'scene' in config:
            scene = config.pop('scene')
            assert scene in self.envs.keys()
        else:
            scene = 'roundabout' # default
        self.env = self.envs[scene](config)

        self.render_mode = render_mode
        self.action_space = convert_old_gym_space_to_gymnasium_space(self.env.action_space)
        # self.action_space = convert_old_gym_space_to_gymnasium_space(self.env.action_space[])
        # print(self.env.observation_space)
        self.observation_space = convert_old_gym_space_to_gymnasium_space(self.env.observation_space)
        self.metadata = getattr(self.env, "metadata", {"render_modes": []})
        self.reward_range = getattr(self.env, "reward_range", None)
        self.spec = getattr(self.env, "spec", None)

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        # --- metadrive --
        self.engine = self.env.engine
        # self.vehicles = self.env.vehicles
        # self.current_track_vehicle = self.env.current_track_vehicle

        if seed is not None:
            self.env.seed(seed)
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated_agent = obs.keys() - done.keys()
        for a in terminated_agent:
            obs.pop(a)
        if self.render_mode == "human":
            self.render()
        # obs = metadrive_dict_to_OrderedDict(obs)
        return metadrive_to_terminated_truncated_step_api((obs, reward, done, info))
  
    def render(self, mode=None, **kwargs) -> Any:
        if mode == None:
            mode = "top_down"
            if kwargs == None:
                kwargs = {"film_size": (1000, 1000)} # TODO: find best size for each scene
        self.env.render(mode=mode, **kwargs)

    def close(self):
        self.env.close()

    @property
    def vehicles(self):
        """
        Return all active vehicles
        :return: Dict[agent_id:vehicle]
        """
        return self.env.agent_manager.active_agents

    def __str__(self):
        """Returns the wrapper name and the unwrapped environment string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)




if __name__ == "__main__":
    import time
    import numpy as np
    from colorlog import logger

    render_on_mac = False
    expert_takeover = False
    print_epi_info = False
    pygame_render = True

    config = dict(
            scene='roundabout',
            use_render=False,
            num_agents=2,
            # "manual_control": True,
            # crash_done=True,
            # "agent_policy": ManualControllableIDMPolicy
            return_single_space=True,
            )
    
    env = RLlibMultiAgentMetaDrive(config)
    o, info = env.reset()


    # logger.debug("env step return o", type(o))
    # print(o)

    print(type(env.observation_space))
    print(type(env.action_space))
    # print(type(env.env.action_space))
    # a = env.action_space.sample()
    # old_a = env.env.action_space.sample()
    # print(a)
    # print(old_a)
    # o_sample = env.observation_space.sample()
    # logger.debug("obs_space sampled o", type(o_sample))
    # print(o_sample)
    # print(type(s))
    # exit()
    done = False
    actions = {}
    s = 0
    while not done:
        for a in o:
            actions[a] = [0, 0.5]
        o, r, terminated, truncated, info = env.step(actions)
        print(type(o))
        print(o.keys())
        print(type(r))
        print(r.keys())
        print(type(terminated))
        print(terminated.keys())
        print(type(truncated))
        print(truncated.keys())
        exit()
        s += 1
        done = terminated['__all__']
        # print(r)
        # print('>>> o:', o.keys())
        # print('>>> env.action space:', env.action_space.keys())
        # print('>>> env.vehicle:', env.vehicles.keys())
        # print(info)
        # print('\n')
        
        env.render(mode="top_down", film_size=(1000, 1000))
    print(s) 
    env.close()