import numpy as np
from gymnasium.wrappers import EnvCompatibility
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete

from ray import air, tune
from ray.tune.registry import register_trainable, register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.utils.gym import convert_old_gym_space_to_gymnasium_space, check_old_gym_env

try:
    from env.metadrive_env import RLlibMetaDriveEnv as MetaDriveEnv
except:
    from .env.metadrive_env import RLlibMetaDriveEnv as MetaDriveEnv
    
# from env.metadrive_ import SubMetaDriveEnv as MetaDriveEnv

config = dict(
        use_render=False,
        traffic_density=0.1,
        environment_num=10,
        random_agent_model=False,
        random_lane_width=False,
        random_lane_num=False,
        map=7,  # seven block
        horizon=1000,
        # start_seed=np.random.randint(0, 1000),
        start_seed=np.random.randint(5000),
        # ===== Reward Scheme =====
        # success_reward=10.0,
        # out_of_road_penalty=10.0, #5.0,
        # crash_vehicle_penalty=10.0, #5.0,
        # crash_object_penalty=10.0, #5.0,
        # driving_reward=1.0, #1.0,
        # speed_reward=0.1,
        # use_lateral_reward=False,
    )

def env_creator(env_config):
    return MetaDriveEnv(env_config)
     

register_env("MetaDriveEnv", env_creator)

# env = env_creator(config)

ppo_config = (
    PPOConfig()
    .framework('torch')
    .resources(num_gpus=0)
    .rollouts(
        num_rollout_workers=4, 
        rollout_fragment_length=200
    )
    .training(
        train_batch_size=8000,
        gamma=0.99,
        lr=5e-5,
        sgd_minibatch_size=100,
        num_sgd_iter=20,
        lambda_=0.95,
    )
    # .evaluation(
    #     evaluation_interval=2,
    #     evaluation_duration=40,
    #     evaluation_config=dict(env_config=dict(environment_num=200, start_seed=0)),
    #     evaluation_num_workers=1,)
    .environment(env="MetaDriveEnv", render_env=False, env_config=config, disable_env_checking=True)
)

# print(pretty_print(ppo_config.to_dict()))
# env.close()
# algo = ppo_config.build()
stop = {
        "training_iteration": 1000,
        "timesteps_total": 1e7,
        "episode_reward_mean": 1000,
}

results = tune.Tuner(
            "PPO",
            param_space=ppo_config, 
            run_config=air.RunConfig(stop=stop, verbose=1)
        ).fit()

# print(pretty_print(results))
print(results)


# for i in range(1000):
#     result = algo.train()
#     print(pretty_print(result))

#     if i % 10 == 0:
#         checkpoint_dir = algo.save()
#         print(f"Checkpoint saved in directory {checkpoint_dir}")