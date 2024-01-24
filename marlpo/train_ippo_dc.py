from ray import tune

import math

from algo.algo_ippo_dc import IPPODCConfig, IPPODCTrainer
from callbacks import MultiAgentDrivingCallbacks
from env.env_wrappers import get_rllib_compatible_env, get_neighbour_env
from env.env_utils import get_metadrive_ma_env_cls
from train import train
from utils.utils import (
    get_train_parser, 
    get_training_resources, 
    get_other_training_configs,
)

ALGO_NAME = "IPPO-DC"

TEST = True # <~~ Default TEST mod here! Don't comment out this line!
            # Also can be assigned in terminal command args by "--test"
            # Will be True once anywhere (code/command) appears True!
TEST = False # <~~ Comment/Uncomment to use TEST/Training mod here! 

SCENE = "intersection" # <~~ Change env name here!
# it will be automaticlly converted to env class
# intersection roundabout


SEEDS = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]


NUM_AGENTS = None # <~~ set to None for env's default num_agents
NEI_DISTANCE = 40
NUM_NEIGHBOURS = 4


EXP_DES = "phi=1_(3)"

if __name__ == "__main__":
    # === Get & Check for Args ===
    args = get_train_parser().parse_args()
    NUM_AGENTS, exp_name, n_rollout_workers, n_cpus_per_worker, SEEDS, TEST = \
                                                    get_other_training_configs(
                                                        args=args,
                                                        algo_name=ALGO_NAME, 
                                                        exp_des=EXP_DES, 
                                                        scene=SCENE, 
                                                        num_agents=NUM_AGENTS,
                                                        seeds=SEEDS,
                                                        test=TEST) 

    # === Get Environment ===
    env_name, env_cls = \
            get_rllib_compatible_env(
            get_neighbour_env(
            get_metadrive_ma_env_cls(SCENE)), return_class=True)

    # === Environmental Configs ===
    env_config = dict(
        num_agents=NUM_AGENTS[0] if isinstance(NUM_AGENTS, list) else NUM_AGENTS,
        return_single_space=True,
        start_seed=tune.grid_search(SEEDS),
        # == neighbour config ==
        neighbours_distance=NEI_DISTANCE,
    )

# ╭──────────────── for test ─────────────────╮
    stop = {"timesteps_total": 2e6}            
    if TEST : stop ={"training_iteration": 5}    
# ╰───────────────────────────────────────────╯

    # === Algo Configs ===
    algo_config = (
        IPPODCConfig()
        .framework('torch')
        .resources(
            num_cpus_per_worker=n_cpus_per_worker,
            num_gpus=0,
        )
        .rollouts(
            num_rollout_workers=n_rollout_workers,
        )
        .callbacks(MultiAgentDrivingCallbacks)
        .multi_agent(
        )
        .training(
            train_batch_size=1024,
            gamma=0.99,
            lr=3e-5,
            # lr=3e-4,
            # lr=tune.grid_search([3e-4, 3e-5]),
            sgd_minibatch_size=512,
            num_sgd_iter=5,
            lambda_=0.95,
            model=dict(
                fcnet_activation='tanh',
            ),
            entropy_coeff=0, # 默认为0 
            vf_loss_coeff=1,
            kl_coeff=tune.grid_search([0]), # 约束新旧策略更新, 默认为0.2 
            _enable_learner_api=False,
        )
        .rl_module(_enable_rl_module_api=False)
        .environment(
            env=env_name, 
            render_env=False, 
            env_config=env_config, 
            disable_env_checking=True
        )
        .update_from_dict(dict(
            nei_rewards_mode='mean', 
            nei_reward_if_no_nei='self',
            phi=tune.grid_search([1]),
            # norm_adv=False,
            norm_adv=tune.grid_search([False]),
            # == IPPO ==
            old_value_loss=True,
            vf_clip_param=100,
            # == Env ==
            num_neighbours=NUM_NEIGHBOURS, # max num of neighbours in use
        ))
    )

    # === Launch training ===
    results = train(
        IPPODCTrainer,
        config=algo_config,
        stop=stop,
        exp_name=exp_name,
        checkpoint_freq=3,
        keep_checkpoints_num=5,
        num_gpus=0,
        results_path='exp_'+ALGO_NAME,
        test_mode=TEST,
    )
