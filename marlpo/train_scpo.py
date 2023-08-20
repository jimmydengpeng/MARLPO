from train import train
from ray import tune

import math

from algo.algo_scpo import SCPOConfig, SCPOTrainer
from callbacks import MultiAgentDrivingCallbacks
from env.env_wrappers import get_rllib_compatible_env, get_neighbour_env
from env.env_utils import get_metadrive_ma_env_cls
from utils import (
    get_train_parser, 
    get_training_resources, 
    get_other_training_configs,
)
from utils.debug import printPanel


TEST = False # <~~ Toggle TEST mode here! 
# TEST = True

ALGO_NAME = "SCPO"
SCENE = "intersection" if not TEST else "intersection" 

# === Env Seeds ===
# seeds = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
# seeds = [5000, 6000, 7000]
SEEDS = [5000]

# NUM_AGENTS = [30]
NUM_AGENTS = 30
NEI_DISTANCE = 40
NUM_NEIGHBOURS = 4
EXP_DES = "BEST(0,1.25)(10,0)_1"
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
        neighbours_distance=NEI_DISTANCE,
    )

# ╭──────────────── for test ─────────────────╮
    stop = {"timesteps_total": 1.2e6}            
    if TEST : stop ={"training_iteration": 5}    
# ╰───────────────────────────────────────────╯

    # === Algo Configs ===
    algo_config = (
        SCPOConfig()
        .framework('torch')
        .resources(
            num_cpus_per_worker=0.125,
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
                fcnet_activation='tanh',
            ),
            entropy_coeff=0, # 默认为0 
            vf_loss_coeff=1,
            kl_coeff=0, # 约束新旧策略更新 # 默认为0.2 
            _enable_learner_api=False,
        )
        .rl_module(_enable_rl_module_api=False)
        .environment(env=env_name, render_env=False, env_config=env_config, disable_env_checking=True)
        .update_from_dict(dict(
            # == Policy Shifting ==
            idv_clip_param=0.5, # no use
            team_clip_param=0.5, # TODO: tuning
            idv_kl_coeff_schedule=[
                (0, 0), 
                (tune.grid_search([0.8*1e6, 0.9*1e6, 1*1e6]), 1.5)
            ],
            team_kl_coeff_schedule=[
                (0, 10), 
                (tune.grid_search([0.9*1e6, 1*1e6, 1.1*1e6]), 0)
            ],
            # == SVO ==
            use_svo=False, # whether to use svo-reward, if False, use original reward
            fixed_svo=math.pi/4, # svo value if use_svo
            nei_rewards_mode='mean', 
            nei_reward_if_no_nei='self',
            nei_rewards_add_coeff=1, # when use_svo=False
            norm_adv=True,
            # == Common ==
            old_value_loss=True,
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
        checkpoint_score_attribute='SuccessRate',
        keep_checkpoints_num=10,
        num_gpus=0,
        results_path='exp_SCPO',
        test_mode=TEST,
    )

    best_res = results.get_best_result(metric="SuccessRate", mode="max").config
    printPanel({
        'idv_kl_coeff_schedule': best_res['idv_kl_coeff_schedule'], 
        'idv': f"{best_res['idv_kl_coeff_schedule'][-1][0]/1e6} • 1e6",
        'team_kl_coeff_schedule': best_res['team_kl_coeff_schedule'], 
        'team': f"{best_res['team_kl_coeff_schedule'][-1][0]/1e6} • 1e6",
    }, title='Best kl_coeff_schedule')
