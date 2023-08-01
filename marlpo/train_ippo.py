from ray import tune

from algo import IPPOConfig, IPPOTrainer
from env.env_wrappers import get_rllib_compatible_env, get_neighbour_md_env
from env.env_utils import get_metadrive_ma_env_cls
from callbacks import MultiAgentDrivingCallbacks

from train import train
from utils import (
    get_train_parser, 
    get_training_resources, 
    get_other_training_configs,
)


TEST = False
# TEST = True

ALGO_NAME = "IPPO"
SCENE = "intersection" if not TEST else "intersection" 

# seeds = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
SEEDS = [5000]

NUM_AGENTS = 4

EXP_DES = "(obs=91)(ind)"
INDEPENDT = True


if __name__ == "__main__":
    args = get_train_parser().parse_args()
    TEST = TEST or args.test
    NUM_AGENTS = args.num_agents if args.num_agents else NUM_AGENTS
    MA_CONFIG = None if not INDEPENDT else dict(
                policies=set([f"agent{i}" for i in range(NUM_AGENTS)]),
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id))
    # if TEST 
    stop, exp_name, n_rlt_workers, seeds = get_other_training_configs(
                                                algo_name=ALGO_NAME, 
                                                exp_des=EXP_DES, 
                                                scene=SCENE, 
                                                num_agents=NUM_AGENTS,
                                                seeds=SEEDS, 
                                                test=TEST) 
    # === Environment ===
    env_name, env_cls = get_rllib_compatible_env(
                            get_metadrive_ma_env_cls(SCENE), 
                            return_class=True)

    # === Environmental Setting ===
    env_config = dict(
        use_render=False,
        num_agents=NUM_AGENTS,
        allow_respawn=tune.grid_search([(not INDEPENDT)]),
        return_single_space=True,
        start_seed=tune.grid_search(seeds),
        vehicle_config=dict(
            lidar=dict(
                num_lasers=72, 
                distance=40, 
                num_others=tune.grid_search([0]),
        )))

    # ========== changable =========== 
    stop = {"timesteps_total": 2e6}
    if args.num_workers:
        n_rlt_workers = args.num_workers
    if TEST:
        stop = {"training_iteration": 1}
        # env_config['num_agents'] = 30
    # ================================ 

    # === Algo Setting ===
    ppo_config = (
        IPPOConfig()
        .framework('torch')
        .resources(
            **get_training_resources()
        )
        .rollouts(
            num_rollout_workers=n_rlt_workers,
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
            **MA_CONFIG,
        )
        .environment(
            env=env_name,
            render_env=False,
            env_config=env_config,
            disable_env_checking=True,
        )
    )


    # === Launch training ===
    # 使用CoPO修改过的PPO算法(IPPO)
    train(
        IPPOTrainer,
        config=ppo_config,
        stop=stop,
        exp_name=exp_name,
        checkpoint_freq=10,
        keep_checkpoints_num=3,
        num_gpus=0,
        results_path='exp_SoCO',
        test_mode=TEST,
    )
