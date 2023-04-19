
from ray import air, tune
from ray.tune.registry import register_trainable, register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.utils.gym import convert_old_gym_space_to_gymnasium_space, check_old_gym_env

from callbacks import MultiAgentDrivingCallbacks

try:
    from env.multi_agent_metadrive import RLlibMultiAgentMetaDrive as MultiAgentMetaDriveEnv
except:
    from .env._multi_agent_metadrive import RLlibMultiAgentMetaDrive as MultiAgentMetaDriveEnv
    

def env_creator(env_config):
    return MultiAgentMetaDriveEnv(env_config)

register_env("MultiAgentMetaDriveEnv", env_creator)



def train(scene, num_agent, training_epoch=10000):

    env_config = dict(
            scene=scene,
            # scene='bottleneck',
            # scene='tollgate',
            use_render=False,
            num_agents=num_agent,
            return_single_space=True,
            # "manual_control": True,
            # crash_done=True,
            # "agent_policy": ManualControllableIDMPolicy
        )

    ppo_config = (
        PPOConfig()
        .framework('torch')
        .resources(num_gpus=0)
        .rollouts(
            num_rollout_workers=4,
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
        .environment(env="MultiAgentMetaDriveEnv", render_env=False, env_config=env_config, disable_env_checking=False)
    )

    # print(pretty_print(ppo_config.to_dict()))
    # env.close()
    algo = ppo_config.build()
    stop = {
            # "training_iteration": 1000,
            "timesteps_total": 1e6,
            # "episode_reward_mean": 1000,
    }
    
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


    for i in range(training_epoch):
        result = algo.train()
        try:
            print(pretty_print(result))
        except:
            print(result)

        if (i+1) % 100 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")


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
    for scene in scenes:
    # scene = 'roundabout'
        train(scene, default_agents[scene])
