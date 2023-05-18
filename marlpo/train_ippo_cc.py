from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv
from ray import tune

from ray.rllib.algorithms.ppo import PPOConfig

from metadrive import (
    MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv, MultiAgentTollgateEnv,
    MultiAgentBottleneckEnv, MultiAgentParkingLotEnv
)

# from copo.torch_copo.algo_ippo import IPPOTrainer
# from copo.torch_copo.utils.callbacks import MultiAgentDrivingCallbacks
from marlpo.algo_ippo import IPPOTrainer
from marlpo.train.train import train
from marlpo.env.env_wrappers import get_rllib_compatible_new_gymnasium_api_env
# from copo.torch_copo.utils.utils import get_train_parser
from marlpo.callbacks import MultiAgentDrivingCallbacks
from marlpo.utils.utils import get_other_training_resources

TEST = False
# TEST = True
# SCENE = "intersection"
SCENE = "roundabout"

# seeds = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
seeds = [5000]

if __name__ == "__main__":
    # ===== Environment =====
    scenes = {
        "roundabout": MultiAgentRoundaboutEnv,
        "intersection": MultiAgentIntersectionEnv,
        "tollgate": MultiAgentTollgateEnv,
        "bottleneck": MultiAgentBottleneckEnv,
        "parkinglot": MultiAgentParkingLotEnv,
    }

    ''' for Multi-Scene training at once! '''
    # We can grid-search the environmental parameters!
    # envs = tune.grid_search([
    #     get_rllib_compatible_new_gymnasium_api_env(MultiAgentRoundaboutEnv),
    #     get_rllib_compatible_new_gymnasium_api_env(MultiAgentIntersectionEnv),
    #     get_rllib_compatible_new_gymnasium_api_env(MultiAgentTollgateEnv),
    #     get_rllib_compatible_new_gymnasium_api_env(MultiAgentBottleneckEnv),
    #     get_rllib_compatible_new_gymnasium_api_env(MultiAgentParkingLotEnv),
    # ])

    if TEST: SCENE = "roundabout" 

    env = get_rllib_compatible_new_gymnasium_api_env(scenes[SCENE])

    # === Environmental Setting ===
    num_agents = 4
    env_config = dict(
        use_render=False,
        num_agents=num_agents,
        return_single_space=True,
        # "manual_control": True,
        # crash_done=True,
        # "agent_policy": ManualControllableIDMPolicy
        # start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000])
        start_seed=tune.grid_search(seeds)
    )
    if TEST:
        env_config["start_seed"] = 5000

    if TEST:
        stop = {"training_iteration": 1}
        exp_name = "TEST"
        num_rollout_workers = 1
    else:
        stop = {
            # "episodes_total": 60000,
            "timesteps_total": 1e6,
            # "episode_reward_mean": 1000,
        }
        if len(seeds) == 1:
            exp_name = f"IPPO_CC_{SCENE.capitalize()}_seed={seeds[0]}_{num_agents}agents"
        else:
            exp_name = f"IPPO__CC_{SCENE.capitalize()}_{len(seeds)}seeds_{num_agents}agents"

        num_rollout_workers = 6
    

    # ===== Algo Setting =====

    ppo_config = (
        PPOConfig()
        .framework('torch')
        .resources(
            **get_other_training_resources()
        )
        .rollouts(
            num_rollout_workers=num_rollout_workers,
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
        .environment(env=env, render_env=False, env_config=env_config, disable_env_checking=False)
    )


    # === Launch training ===
    '''
    使用CoPO修改过的PPO算法
    '''
    train(
        IPPOTrainer,
        config=ppo_config,
        stop=stop,
        exp_name=exp_name,
        checkpoint_freq=10,
        keep_checkpoints_num=3,
        num_gpus=0,
        test_mode=TEST,
    )
