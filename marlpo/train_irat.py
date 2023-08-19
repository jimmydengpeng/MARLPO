from train import train
from ray import tune

import math

from algo.algo_irat import IRATConfig, IRATTrainer
from callbacks import MultiAgentDrivingCallbacks
from env.env_wrappers import get_rllib_compatible_env, get_neighbour_env
from env.env_utils import get_metadrive_ma_env_cls
from utils import (
    get_train_parser, 
    get_training_resources, 
    get_other_training_configs,
)


TEST = False # <~~ Toggle TEST mode here! 
# TEST = True

ALGO_NAME = "IRAT"
SCENE = "intersection" if not TEST else "intersection" 

# === Env Seeds ===
# seeds = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
# seeds = [5000, 6000, 7000]
SEEDS = [5000]

# NUM_AGENTS = [30]
NUM_AGENTS = 30
NEI_DISTANCE = 40
NUM_NEIGHBOURS = 4
# EXP_DES = "v<idv:0->0.2, team:1->0.8>"
EXP_DES = "BEST(0,1.25)(10,0)_7"
# EXP_DES = "v1<kl_coeff><(0,0.5)(1, 0.5)>"

if __name__ == "__main__":
    # === Get Args ===
    args = get_train_parser().parse_args()
    NUM_AGENTS, exp_name, num_rollout_workers, SEEDS, TEST = \
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
        num_agents=tune.grid_search(NUM_AGENTS),
        return_single_space=True,
        start_seed=tune.grid_search(SEEDS),
        delay_done=25,
        vehicle_config=dict(
            lidar=dict(
                num_lasers=72, 
                distance=40, 
                num_others=0)),
        # == neighbour config ==
        add_nei_state=False, 
        use_dict_obs=False,
        add_compact_state=False,
        neighbours_distance=NEI_DISTANCE,
    )

    # ────────────── for test ────────────── # 
    stop = {"timesteps_total": 1.5e6}            
    if TEST : stop ={"training_iteration": 1}    
    # ────────────────────────────────────── # 

    # === Algo Configs ===
    algo_config = (
        IRATConfig()
        .framework('torch')
        .resources(
            num_cpus_per_worker=0.25,
            **get_training_resources()
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
            sgd_minibatch_size=512,
            num_sgd_iter=5,
            lambda_=0.95,
            model=dict(
                # == RLlib built-in attention net config ==
                use_attention=False,
                max_seq_len=10,
                attention_num_transformer_units=1,
                attention_num_heads=2,
                attention_head_dim=32,
                attention_position_wise_mlp_dim=32,
                custom_model_config=dict(
                ),
                free_log_std=False,
                fcnet_activation='tanh',
            ),
            # use_kl_loss=False, # for single ppo's kl(pi_old|pi_new)
            entropy_coeff=0, # 不能加上最大化熵? # 默认为0 
            # grad_clip=40,
            # grad_clip_by='norm',
            vf_loss_coeff=1,
            kl_coeff=0, # 约束新旧策略更新 # 默认为0.2 
            # kl_target=0,
            _enable_learner_api=False,
        )
        .rl_module(_enable_rl_module_api=False)
        .environment(env=env_name, render_env=False, env_config=env_config, disable_env_checking=True)
        .update_from_dict(dict(
            # == IRAT ==
            idv_policy_no_team=True,
            idv_clip_param=0.5,
            team_clip_param=0.5,
            # idv_kl_coeff=0.2,
            # idv_kl_end_coeff=0.5,
            idx_kl_coeff_schedule=[
                (0, 0), 
                (0.85*NUM_AGENTS[0]*1.5e6, 1)
            ],
            team_kl_coeff_schedule=[
                (0, 10), 
                (tune.grid_search([1.15*NUM_AGENTS[0]*1.5e6, 1.25*NUM_AGENTS[0]*1.5e6]), 0)
            ],
            # team_kl_coeff=0.2,
            # team_kl_end_coeff=0.5,
            # == SaCo ==
            use_svo=True, #tune.grid_search([True, False]), # whether or not to use svo to change reward, if False, use original reward
            use_fixed_svo=False,
            fixed_svo=math.pi/4, #tune.grid_search([math.pi/4, math.pi/6, math.pi/3]),
            svo_init_value='0', # in [ '0', 'pi/4', 'pi/2', 'random' ]
            svo_mode='full', #tune.grid_search(['full', 'restrict']),
            # nei_rewards_mode='mean_nei_rewards', #'attentive_one_nei_reward',
            nei_rewards_mode='mean', #'attentive_one_nei_reward',
            nei_reward_if_no_nei='self',
            nei_rewards_add_coeff=1,
            norm_adv=True,
            # == Common ==
            huber_value_loss=True,
            # old_value_loss=True,
            num_neighbours=NUM_NEIGHBOURS, # the max num of neighbours in use
            # == CC ==
            use_central_critic=False,
            counterfactual=False,
            fuse_mode="none",
        ))
    )

    # === Launch training ===
    train(
        IRATTrainer,
        config=algo_config,
        stop=stop,
        exp_name=exp_name,
        checkpoint_freq=5,
        keep_checkpoints_num=5,
        num_gpus=0,
        results_path='exp_IRAT',
        test_mode=TEST,
    )
