from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv
from ray import tune

from ray.rllib.algorithms.ppo import PPOConfig

from metadrive import (
    MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv, MultiAgentTollgateEnv,
    MultiAgentBottleneckEnv, MultiAgentParkingLotEnv
)

# from copo.torch_copo.algo_ippo import IPPOTrainer
# from copo.torch_copo.utils.callbacks import MultiAgentDrivingCallbacks
from marlpo.algo_arccppo import ARCCPPOConfig, ARCCPPOTrainer, get_ccppo_env

from marlpo.train.train import train
from marlpo.env.env_wrappers import get_rllib_compatible_new_gymnasium_api_env
# from copo.torch_copo.utils.utils import get_train_parser
from marlpo.callbacks import MultiAgentDrivingCallbacks
from marlpo.utils.utils import get_other_training_resources, get_num_workers


# CCPPO_CONFIG = None


TEST = False # <~~ Toggle TEST mod here! 
TEST = True

# === Training Scene ===
SCENE = "roundabout"
# SCENE = "intersection"

if TEST: SCENE = "roundabout" 

# === Env Seeds ===
# seeds = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
seeds = [5000]
EXP_SUFFIX = "_V1"

if __name__ == "__main__":
    # === Environment ===
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

    # ccppo
    env = get_ccppo_env(scenes[SCENE])

    # ===== Environmental Setting =====

    num_agents = 8 #, 8, 16, 32, 40]
    env_config = dict(
        num_agents=num_agents,
        return_single_space=True,
        start_seed=tune.grid_search(seeds),
        neighbours_distance=10,
    )
 
    if TEST:
        env_config["start_seed"] = 5000
        stop = {"training_iteration": 1}
        exp_name = "TEST"
        num_rollout_workers = 0
    else:
        stop = {"timesteps_total": 1e6}
        if len(seeds) == 1:
            exp_name = f"ARCCPPO_{SCENE.capitalize()}_seed={seeds[0]}_{num_agents}agents"+EXP_SUFFIX
        else:
            exp_name = f"ARCCPPO_{SCENE.capitalize()}_{len(seeds)}seeds_{num_agents}agents"+EXP_SUFFIX

        num_rollout_workers = get_num_workers()
    

    # === Algo Setting ===

    ppo_config = (
        ARCCPPOConfig()
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
            model={
                "custom_model_config": {
                    "num_neighbours": 4,
                }
            },
        )
        # .multi_agent(
        # )
        # .evaluation(
        #     evaluation_interval=2,
        #     evaluation_duration=40,
        #     evaluation_config=dict(env_config=dict(environment_num=200, start_seed=0)),
        #     evaluation_num_workers=1,)
        .environment(env=env, render_env=False, env_config=env_config, disable_env_checking=False)
        .update_from_dict(dict(
            counterfactual=tune.grid_search([True, False]),
            # fuse_mode="concat",
            # fuse_mode=tune.grid_search(["mf"]),
            fuse_mode=tune.grid_search(["mf", "concat", "none"]),
            random_order=True,
        ))
    )

    # === Launch training ===
    train(
        ARCCPPOTrainer,
        config=ppo_config,
        stop=stop,
        exp_name=exp_name,
        checkpoint_freq=10,
        keep_checkpoints_num=3,
        num_gpus=0,
        test_mode=TEST,
    )
