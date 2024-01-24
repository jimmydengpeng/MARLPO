from ray import tune

import math

from algo.algo_scpo import SCPOConfig, SCPOTrainer
from callbacks import MultiAgentDrivingCallbacks
from env.env_wrappers import get_rllib_compatible_env, get_neighbour_env
from env.env_utils import get_metadrive_ma_env_cls
from train import train
from utils.utils import (
    get_train_parser, 
    get_training_resources, 
    get_other_training_configs,
)

ALGO_NAME = "SCPO"

TEST = True # <~~ Default TEST mod here! Don't comment out this line!
            # Also can be assigned in terminal command args by "--test"
            # Will be True once anywhere (code/command) appears True!
TEST = False # <~~ Comment/Uncomment to use TEST/Training mod here! 

SCENE = "roundabout" # <~~ Change env name here!
# it will be automaticlly converted to env class
# intersection 


SEEDS = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
# SEEDS = [9000, 10000, 11000, 12000]
# SEEDS = [5000, 6000, 7000, 8000]
# SEEDS = [5000]
# SEEDS = [5000, 6000, 7000, 8000]
# SEEDS = [5000, 9000]
# SEEDS = [9000, 10000, 11000, 12000]

NUM_AGENTS = None # <~~ set to None for env's default num_agents
NEI_DISTANCE = 40
NUM_NEIGHBOURS = 4


EXP_DES = "2M_lr3e-5_[main](0->1,5->1)"

if __name__ == "__main__":
    # === Get & Check for Args ===
    args = get_train_parser().parse_args()
    NUM_AGENTS, exp_name, num_rollout_workers, num_cpus_per_worker, SEEDS, TEST = \
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
                get_metadrive_ma_env_cls(SCENE)), 
            return_class=True)

    # === Environmental Configs ===
    env_config = dict(
        num_agents=NUM_AGENTS[0] if isinstance(NUM_AGENTS, list) else NUM_AGENTS,
        return_single_space=True,
        start_seed=tune.grid_search(SEEDS),
        # == neighbour config ==
        neighbours_distance=NEI_DISTANCE,
    )
    # env_config.update({'num_agents': tune.grid_search(num_agents) 
    #                    if len(num_agents) > 1 else num_agents[0]})

# ╭──────────────── for test ─────────────────╮
    stop = {"timesteps_total": 2e6}            
    if TEST : stop ={"training_iteration": 5}    
# ╰───────────────────────────────────────────╯

    # === Algo Configs ===
    algo_config = (
        SCPOConfig()
        .framework('torch')
        .resources(
            num_cpus_per_worker=num_cpus_per_worker,
            num_gpus=0,
        )
        .rollouts(
            num_rollout_workers=num_rollout_workers,
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
            entropy_coeff=0, # 默认为0 # TODO: tunnable
            vf_loss_coeff=1,
            kl_coeff=0, # 约束新旧策略更新, 默认为0.2 
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
            # == Policy Shifting ==
            idv_clip_param=0.5, # no use
            # team_clip_param=0.5,
            team_clip_param=tune.grid_search([0.3]),
            idv_kl_coeff_schedule=[
                (0, 0), 
                (2e6, 1)
            ],
            team_kl_coeff_schedule=[
                (0, 5), 
                (2e6, 1)
            ],
            # == SVO ==
            use_svo=False, # whether to use svo-reward, if False, use original reward
            fixed_svo=math.pi/4, # svo value if use_svo
            nei_rewards_mode='mean', 
            nei_reward_if_no_nei='self',
            nei_rewards_add_coeff=tune.grid_search([1]), # when use_svo=False
            norm_adv=True,
            # == Common ==
            old_value_loss=True,
            vf_clip_param=100,
            num_neighbours=NUM_NEIGHBOURS, # max num of neighbours in use
        ))
    )

    # === Launch training ===
    results = train(
        SCPOTrainer,
        config=algo_config,
        stop=stop,
        exp_name=exp_name,
        checkpoint_freq=3,
        keep_checkpoints_num=5,
        num_gpus=0,
        results_path='exp_'+ALGO_NAME,
        test_mode=TEST,
    )
