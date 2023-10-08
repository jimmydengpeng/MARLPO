from ray import tune

from algo.algo_ippo import IPPOConfig, IPPOTrainer
from env.env_wrappers import get_rllib_compatible_env, get_tracking_md_env
from env.env_utils import get_metadrive_ma_env_cls
from callbacks import MultiAgentDrivingCallbacks

from train import train
from utils.utils import (
    get_train_parser, 
    get_training_resources, 
    get_other_training_configs,
)

''' Training Command Exmaple:
python marlpo/train_ippo.py --num_agents=30 --num_workers=4 --test
'''
ALGO_NAME = "IPPO"

TEST = True # <~~ Default TEST mod here! Don't comment out this line!
            # Also can be assigned in terminal command args by "--test"
            # Will be True once anywhere (code/command) appears True!
# TEST = False # <~~ Comment to use TEST mod here! Uncomment to use training mod!

SCENE = "intersection" # <~~ Change env name here! will be automaticlly converted to env class

SEEDS = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
SEEDS = [5000, 6000, 7000]
SEEDS = [8000, 9000, 10000, 11000, 12000]
SEEDS = [5000]

NUM_AGENTS = 30

EXP_DES = ""
INDEPENDT = False


if __name__ == "__main__":
    # == Get Args ==
    args = get_train_parser().parse_args()
    MA_CONFIG = {} if not INDEPENDT else dict(
                policies=set([f"agent{i}" for i in range(NUM_AGENTS)]),
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id))

    # Check for all args & configs!
    num_agents, exp_name, num_rollout_workers, seeds, TEST = \
                                    get_other_training_configs(
                                        args=args,
                                        algo_name=ALGO_NAME, 
                                        exp_des=EXP_DES, 
                                        scene=SCENE, 
                                        num_agents=NUM_AGENTS,
                                        seeds=SEEDS,
                                        test=TEST) 
    # === Get Environment ===
    env_name, env_cls = get_rllib_compatible_env(
                            get_tracking_md_env(
                                get_metadrive_ma_env_cls(SCENE)
                            ), return_class=True)

    # === Environmental Configs ===
    env_config = dict(
        allow_respawn=(not INDEPENDT),
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


    # === Algo Setting ===
    ppo_config = (
        IPPOConfig()
        .framework('torch')
        .resources(
            num_cpus_per_worker=0.125,
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
        .multi_agent(**MA_CONFIG)
        .environment(
            env=env_name,
            render_env=False,
            env_config=env_config,
            disable_env_checking=True,
        )
        .update_from_dict(dict(
            norm_adv=False,
        ))
    )

    # === Launch training ===
    # 使用CoPO修改过的PPO算法(IPPO)
    train(
        IPPOTrainer,
        config=ppo_config,
        stop=stop,
        exp_name=exp_name,
        checkpoint_freq=3,
        keep_checkpoints_num=3,
        num_gpus=0,
        results_path='exp_'+ALGO_NAME,
        test_mode=TEST,
    )
