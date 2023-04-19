
from ray import air, tune
from ray.tune.registry import register_trainable, register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.utils.gym import convert_old_gym_space_to_gymnasium_space, check_old_gym_env

from callbacks import MultiAgentDrivingCallbacks
from train.train import train

# from env.multi_agent_metadrive import RLlibMultiAgentMetaDrive as MultiAgentMetaDriveEnv
# except:
#     from .env.multi_agent_metadrive import RLlibMultiAgentMetaDrive as MultiAgentMetaDriveEnv
    
from marlpo.env.env_wrappers import get_rllib_compatible_new_gymnasium_api_env

from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv

# def env_creator(env_config):
#     return MultiAgentMetaDriveEnv(env_config)

# register_env("MultiAgentMetaDriveEnv", env_creator)



TEST = False
# TEST = True

    
# env = env_creator(config)

# results = tune.Tuner(
#             "PPO",
#             param_space=ppo_config, 
#             run_config=air.RunConfig(stop=stop, verbose=1)
#         ).fit()

# try:
#     print(pretty_print(results))
# except:
#     print(results)


# for i in range(training_epoch):
#     result = algo.train()
#     try:
#         print(pretty_print(result))
#     except:
#         print(result)

#     if (i+1) % 100 == 0:
#         checkpoint_dir = algo.save()
#         print(f"Checkpoint saved in directory {checkpoint_dir}")

if __name__ == "__main__":
    # for n in (4, 10, 20, 40):
    #     train(n)
    default_agents = dict(
        roundabout=40,
        intersection=30,
        tollgate=40,
        bottleneck=20,
        parkinglot=10,
    )

    scenes = [
        "roundabout",
        "intersection",
        "tollgate",
        "bottleneck",
        "parkinglot",
    ]
    # for scene in scenes:
    # # scene = 'roundabout'
    #     train(scene, default_agents[scene])
    if TEST:
    # Only a compilation test of running waterworld / independent learning.
        stop = {"training_iteration": 1}
        exp_name = "TEST"
        num_rollout_workers = 1
    else:
        stop = {
            # "episodes_total": 60000,
            "timesteps_total": 1e6,
            # "episode_reward_mean": 1000,
        }
        exp_name = "IPPO_Roundabout_8seeds"
        num_rollout_workers = 4
    
    envs = tune.grid_search([
        get_rllib_compatible_new_gymnasium_api_env(MultiAgentRoundaboutEnv),
        get_rllib_compatible_new_gymnasium_api_env(MultiAgentIntersectionEnv),
    ])

    env_config = dict(
            # scene="roundabout",
            # scene='bottleneck',
            # scene='tollgate',
            use_render=False,
            # num_agents=40,
            return_single_space=True,
            # "manual_control": True,
            # crash_done=True,
            # "agent_policy": ManualControllableIDMPolicy
            start_seed=tune.grid_search([8000, 9000,])
            # start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000])
        )

    ppo_config = (
        PPOConfig()
        .framework('torch')
        .resources(num_gpus=0)
        .rollouts(
            num_rollout_workers=num_rollout_workers,
            # rollout_fragment_length=200 # can get from algo?
        )
        .callbacks(MultiAgentDrivingCallbacks)
        .training(
            train_batch_size=1024,
            gamma=0.99,
            lr=3e-4,
            sgd_minibatch_size=512,
            num_sgd_iter=5,
            lambda_=0.95,
        )
        .multi_agent(
        )
        # .evaluation(
        #     evaluation_interval=2,
        #     evaluation_duration=40,
        #     evaluation_config=dict(env_config=dict(environment_num=200, start_seed=0)),
        #     evaluation_num_workers=1,)
        .environment(env=envs, render_env=False, env_config=env_config, disable_env_checking=False)
    )

        # print(pretty_print(ppo_config.to_dict()))
        # env.close()
    # algo = ppo_config.build()




    train(
        "PPO",
        config=ppo_config,
        stop=stop,
        exp_name=exp_name,
        test_mode=TEST,
        checkpoint_freq=10,
        keep_checkpoints_num=3,
    )
