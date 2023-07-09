from ray import tune

from marlpo.algo_sappo import SAPPOConfig, SAPPOTrainer
from marlpo.callbacks import MultiAgentDrivingCallbacks
from marlpo.env.env_wrappers import get_rllib_cc_env
from marlpo.env.env_utils import get_metadrive_ma_env_cls
from marlpo.utils.utils import get_train_parser

from train.train import train
from utils import (
    get_train_parser, 
    get_other_training_resources, 
    get_args_only_if_test,
)



TEST = False # <~~ Toggle TEST mod here! 
# TEST = True

ALGO_NAME = "SAPPO"
SCENE = "intersection" if not TEST else "intersection" 

# === Env Seeds ===
# seeds = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
# seeds = [6000, 7000]
# seeds = [5000]
seeds = [8000, 9000, 10000, 11000, 12000]

# NUM_AGENTS = [4, 8, 16, 30]
NUM_AGENTS = [30]
EXP_DES = "v4(share_vf_5e6)"

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    TEST = TEST or args.test

    # === Environment ===
    env, env_cls = get_rllib_cc_env(get_metadrive_ma_env_cls(SCENE), return_class=True)

    # === Environmental Setting ===
    env_config = dict(
        num_agents=tune.grid_search(NUM_AGENTS),
        return_single_space=True,
        start_seed=tune.grid_search(seeds),
        vehicle_config=dict(lidar=dict(num_lasers=72, distance=40, num_others=0)),
        # == neighbour config ==
        use_dict_obs=True,
        add_compact_state=True, # add BOTH ego- & nei- compact-state simultaneously
        add_nei_state=False,
        num_neighbours=4,
        neighbours_distance=40,
    )

    # if TEST
    stop, exp_name, num_rollout_workers = get_args_only_if_test(algo_name=ALGO_NAME, env_config=env_config, exp_des=EXP_DES, scene=SCENE, num_agents=NUM_AGENTS, test=TEST) # env_config will be modified
    
    stop = {"timesteps_total": 3e6}
    # env_config['num_agents'] = 3


    # === Algo Setting ===
    algo_config = (
        SAPPOConfig()
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
                    "env_cls": env_cls,
                    # 'embedding_hiddens': [64, 64],
                    'attention_dim': 64,
                    "policy_head_hiddens": [64, 64],
                },
                # "vf_share_layers": True,
                "vf_share_layers": True, 
            },
        )
        .environment(env=env, render_env=False, env_config=env_config, disable_env_checking=False)
        .update_from_dict(dict(
            # == Common ==
            num_neighbours=4,
            # == CC ==
            use_central_critic=False,
            counterfactual=False,
            fuse_mode="none",
            # == Attention ==
            use_attention=True,
        ))
    )

    # === Launch training ===
    train(
        SAPPOTrainer,
        config=algo_config,
        stop=stop,
        exp_name=exp_name,
        checkpoint_freq=10,
        keep_checkpoints_num=3,
        num_gpus=0,
        test_mode=TEST,
    )
