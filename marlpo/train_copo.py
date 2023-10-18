from ray import tune

from algo.algo_copo import CoPOConfig, CoPOTrainer
from callbacks import MultiAgentDrivingCallbacks
from env.env_copo import get_lcf_env
from env.env_wrappers import get_rllib_compatible_env
from env.env_utils import get_metadrive_ma_env_cls
from train import train
from utils.utils import (
    get_train_parser, 
    get_training_resources, 
    get_other_training_configs,
)

ALGO_NAME = "CoPO"

TEST = True # <~~ Default TEST mod here! Don't comment out this line!
            # Also can be assigned in terminal command args by "--test"
            # Will be True once anywhere (code/command) appears True!
TEST = False # <~~ Comment/Uncomment to use TEST/Training mod here! 

SCENE ="intersection" #, [roundabout, 'tollgate', 'parkinglot'] # <~~ Change env name here! will be automaticlly converted to env class

SEEDS = [5000]
# SEEDS = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]

NUM_AGENTS = None # <~~ set to None for env's default `num_agents`

EXP_DES = "2M"


if __name__ == "__main__":
    args = get_train_parser().parse_args()
    num_agents, exp_name, num_rollout_workers, seeds, TEST = \
                                    get_other_training_configs(
                                        args=args,
                                        algo_name=ALGO_NAME, 
                                        exp_des=EXP_DES, 
                                        scene=SCENE, 
                                        num_agents=NUM_AGENTS,
                                        seeds=SEEDS,
                                        test=TEST) 
    
    env_name, env_cls = get_rllib_compatible_env(
                        get_lcf_env(
                        get_metadrive_ma_env_cls(SCENE)), return_class=True)
    
    env_config = dict(
        return_single_space=True,
        start_seed=tune.grid_search(seeds),
        vehicle_config=dict(
            lidar=dict(
                num_lasers=72, 
                distance=40, 
                num_others=0,
    )))
    env_config.update({'num_agents': tune.grid_search(num_agents) 
                       if len(num_agents) > 1 else num_agents[0]})

# ╭──────────────── for test ─────────────────╮
    stop = {"timesteps_total": 2e6}            
    if TEST : stop ={"training_iteration": 5}    
# ╰───────────────────────────────────────────╯

    # === Algo Configs ===

    algo_config = (
        CoPOConfig()
        .framework('torch')
        .resources(
            num_cpus_per_worker=0.2,
            **get_training_resources()
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
            _enable_learner_api=False,
        )
        .rl_module(_enable_rl_module_api=False)
        .environment(
            env=env_name,
            render_env=False,
            env_config=env_config,
            disable_env_checking=True,
        )
        .update_from_dict(dict(
            counterfactual=True,
            fuse_mode="concat",
            # fuse_mode=tune.grid_search(["mf", "concat"]),
            mf_nei_distance=10,
        ))
    )

    # === Launch training ===
    train(
        CoPOTrainer,
        config=algo_config,
        stop=stop,
        exp_name=exp_name,
        checkpoint_freq=10,
        keep_checkpoints_num=10,
        num_gpus=0,
        results_path='exp_'+ALGO_NAME,
        test_mode=TEST,
    )
