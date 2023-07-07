from ray import tune

from algo import IPPOConfig, IPPOTrainer
from env import get_rllib_cc_env, get_metadrive_ma_env_cls
from callbacks import MultiAgentDrivingCallbacks

from train import train
from utils import (
    get_train_parser, 
    get_other_training_resources, 
    get_args_only_if_test,
)


TEST = False
# TEST = True

ALGO_NAME = "IPPO"
SCENE = "intersection" if not TEST else "intersection" 

# seeds = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
# seeds = [5000, 6000, 7000]
seeds = [7000]

# NUM_AGENTS = [4, 8, 16, 30]
NUM_AGENTS = [30]

EXP_DES = "(compact-state)"
# EXP_DES = "16a_nei_state"


if __name__ == "__main__":
    args = get_train_parser().parse_args()
    TEST = TEST or args.test

    # === Environment ===
    env, env_cls = get_rllib_cc_env(get_metadrive_ma_env_cls(SCENE), return_class=True)

    # === Environmental Setting ===
    env_config = dict(
        use_render=False,
        num_agents=tune.grid_search(NUM_AGENTS),
        return_single_space=True,
        start_seed=tune.grid_search(seeds),
        vehicle_config=dict(
            lidar=dict(num_lasers=72, distance=40, num_others=0),
            # lidar=dict(num_lasers=tune.grid_search([0]), distance=40, num_others=0),
        ),
        # == neighbour config ==
        use_dict_obs=False,
        add_compact_state=True, # add BOTH ego- & nei- compact-state simultaneously
        add_nei_state=False,
        num_neighbours=tune.grid_search([1, 4]),
        neighbours_distance=20,
        # neighbours_distance=tune.grid_search([10, 20, 30]),
    )

    # if TEST
    stop, exp_name, num_rollout_workers = get_args_only_if_test(algo_name=ALGO_NAME, env_config=env_config, exp_des=EXP_DES, scene=SCENE, num_agents=NUM_AGENTS, test=TEST)
  

    # === Algo Setting ===
    ppo_config = (
        IPPOConfig()
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
        .environment(
            env=env,
            render_env=False,
            env_config=env_config,
            disable_env_checking=False
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
        test_mode=TEST,
    )
