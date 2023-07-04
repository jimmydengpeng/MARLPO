from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv
from ray import tune


from metadrive import (
    MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv, MultiAgentTollgateEnv,
    MultiAgentBottleneckEnv, MultiAgentParkingLotEnv
)

# from copo.torch_copo.algo_ippo import IPPOTrainer
# from copo.torch_copo.utils.callbacks import MultiAgentDrivingCallbacks
from marlpo.algo_arppo import ARCCPPOConfig, ARCCPPOTrainer

from marlpo.train.train import train
from marlpo.env.env_wrappers import get_rllib_cc_env
# from copo.torch_copo.utils.utils import get_train_parser
from marlpo.callbacks import MultiAgentDrivingCallbacks
from marlpo.utils.utils import (
    get_other_training_resources, 
    get_num_workers, 
    get_train_parser, 
    get_abbr_scene,
    get_agent_str
)


TEST = False # <~~ Toggle TEST mod here! 
# TEST = True

SCENE = "intersection" if not TEST else "roundabout" 

# === Env Seeds ===
seeds = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
# seeds = [6000]

NUM_AGENTS = [4, 8, 16, 30]
EXP_DES = "no-nei-navi"

if __name__ == "__main__":
    args = get_train_parser().parse_args()
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
    env = get_rllib_cc_env(scenes[SCENE])

    # ===== Environmental Setting =====

    env_config = dict(
        num_agents=tune.grid_search(NUM_AGENTS),
        return_single_space=True,
        start_seed=tune.grid_search(seeds),
        vehicle_config=dict(
            lidar=dict(num_lasers=tune.grid_search([72]), distance=40, num_others=0),
        ),
        # == neighbour config ==
        neighbour_states=True,
        nei_navi=tune.grid_search([True]),
        # num_neighbours=tune.grid_search([1, 2, 3]),
        num_neighbours=4,
        neighbours_distance=40,
        # neighbours_distance=tune.grid_search([10, 20, 30]),
    )
 
    if TEST:
        env_config["start_seed"] = 5000
        env_config["num_agents"] = 4
        stop = {"training_iteration": 1}
        exp_name = "TEST"
        num_rollout_workers = 0
    else:
        stop = {"timesteps_total": 1e6}
        EXP_SUFFIX = ('_' if EXP_DES else '') + EXP_DES
        exp_name = f"IPPO_{get_abbr_scene(SCENE)}_{get_agent_str(NUM_AGENTS)}agents" + EXP_SUFFIX
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
            # model={
            #     "custom_model": "svo_model",
            #     "custom_model_config": {
            #         "num_neighbours": 4,
            #         "use_attention": True,
            #         "use_central_critic": False,
            #     }
            # },
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
            # == CC ==
            use_central_critic=False,
            # counterfactual=tune.grid_search([False, True]),
            counterfactual=False,
            # fuse_mode=tune.grid_search(["mf", "concat", "none"]),
            # fuse_mode=tune.grid_search(["mf"]),
            fuse_mode="none",
            # == Attention ==
            use_attention=True,
            num_neighbours=4,
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
