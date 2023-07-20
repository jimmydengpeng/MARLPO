from ray import tune

from marlpo.algo_scppo import SCPPOConfig, SCPPOTrainer
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

ALGO_NAME = "SACO"
SCENE = "intersection" if not TEST else "intersection" 

# === Env Seeds ===
# seeds = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
# seeds = [5000, 6000, 7000]
seeds = [5000]
# seeds = [8000, 9000, 10000, 11000, 12000]

# NUM_AGENTS = [4, 8, 16, 30]
# NUM_AGENTS = [8, 16, 30]
# NUM_AGENTS = [30]
NUM_AGENTS = [8]
EXP_DES = "(saco)"

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    TEST = TEST or args.test
    NUM_AGENTS = [args.num_agents] if args.num_agents else NUM_AGENTS

    # === Environment ===
    env, env_cls = get_rllib_cc_env(get_metadrive_ma_env_cls(SCENE), return_class=True)

    # === Environmental Setting ===
    env_config = dict(
        num_agents=tune.grid_search(NUM_AGENTS),
        return_single_space=True,
        start_seed=tune.grid_search(seeds),
        delay_done=25,
        vehicle_config=dict(lidar=dict(num_lasers=72, distance=40, num_others=0)),
        # == neighbour config ==
        use_dict_obs=True,
        add_compact_state=True, # add BOTH ego- & nei- compact-state simultaneously
        add_nei_state=False,
        num_neighbours=5, # determine how many neighbours's abs state will be included in obs
        neighbours_distance=10,
    )

    # if TEST
    stop, exp_name, num_rollout_workers = get_args_only_if_test(algo_name=ALGO_NAME, env_config=env_config, exp_des=EXP_DES, scene=SCENE, num_agents=NUM_AGENTS, test=TEST) # env_config will be modified
    
    stop = {"timesteps_total": 2e6}
    if args.num_workers:
        num_rollout_workers = args.num_workers
    if TEST:
        # stop = {"timesteps_total": 3e6}
        stop = {"training_iteration": 1}
        env_config['num_agents'] = 30


    # === Algo Setting ===
    algo_config = (
        SCPPOConfig()
        .framework('torch')
        .resources(
            **get_other_training_resources()
        )
        .rollouts(
            num_rollout_workers=num_rollout_workers,
        )
        .callbacks(MultiAgentDrivingCallbacks)
        .multi_agent(
        )
        .training(
            train_batch_size=1024,
            # train_batch_size=512,
            # train_batch_size=256,
            gamma=0.99,
            lr=3e-4,
            sgd_minibatch_size=512,
            # sgd_minibatch_size=128,
            # sgd_minibatch_size=64,
            num_sgd_iter=5,
            lambda_=0.95,
            model=dict(
                custom_model_config=dict(
                    env_cls=env_cls,
                    # 'embedding_hiddens': [64, 64],
                    # == Attention ==
                    use_attention=True, # whether use attention backbone or mlp backbone
                    attention_dim=64,
                    attention_heads=4,
                    # == head dim ==
                    policy_head_hiddens=[64, 64],
                    svo_head_hiddens=[64, 64],
                    critic_head_hiddens=[64, 64],
                    # == net arch ==
                    onehot_attention=True,
                    actor_concat_atn_feature=True,
                    critic_concat_svo=True,
                    critic_concat_obs=True, # ──╮ ⛓ better lock together?
                    svo_concat_obs=True,    # ──╯ 
                    # onehot_attention=tune.grid_search([True, False]),
                ),
                free_log_std=True,
            )
        )
        .environment(env=env, render_env=False, env_config=env_config, disable_env_checking=False)
        .update_from_dict(dict(
            # == SaCo == 
            nei_rewards_mode='attentive_one_nei_reward',
            sp_select_mode=tune.grid_search(['bincount', 'numerical']), # only work if use onehot_attention!
            # nei_rewards_mode=tune.grid_search([
            #     'mean_nei_rewards',           # ─╮ 
            #     'max_nei_rewards',            #  │
            #     'nearest_nei_reward',         #  │──> Choose 1 alternatively
            #     'attentive_one_nei_reward',   #  │
            #     # 'attentive_all_nei_reward', # ─╯
            # ]),
            
            # == Common ==
            old_value_loss=True,
            # old_value_loss=True,
            num_neighbours=5,
            # == CC ==
            use_central_critic=False,
            counterfactual=False,
            fuse_mode="none",
        ))
    )

    # === Launch training ===
    train(
        SCPPOTrainer,
        config=algo_config,
        stop=stop,
        exp_name=exp_name,
        checkpoint_freq=10,
        keep_checkpoints_num=3,
        num_gpus=0,
        test_mode=TEST,
    )
