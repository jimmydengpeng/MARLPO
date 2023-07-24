import math
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
seeds = [6000]
# seeds = [8000, 9000, 10000, 11000, 12000]

# NUM_AGENTS = [4, 8, 16, 30]
# NUM_AGENTS = [8, 16, 30]
# NUM_AGENTS = [30]
NUM_AGENTS = [8]
NUM_NEIGHBOURS = 4
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
        use_dict_obs=False,
        add_compact_state=False, # add BOTH ego- & nei- compact-state simultaneously
        add_nei_state=False,
        num_neighbours=NUM_NEIGHBOURS, # determine how many neighbours's abs state will be included in obs
        neighbours_distance=10,
    )

    # if TEST
    stop, exp_name, num_rollout_workers = get_args_only_if_test(algo_name=ALGO_NAME, env_config=env_config, exp_des=EXP_DES, scene=SCENE, num_agents=NUM_AGENTS, test=TEST) # env_config will be modified
    
    stop = {"timesteps_total": 1e6}
    if args.num_workers:
        num_rollout_workers = args.num_workers
    if TEST:
        # stop = {"timesteps_total": 3e6}
        stop = {"training_iteration": 1}
    if TEST and not args.num_agents:
        env_config['num_agents'] = 4


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
                use_attention=tune.grid_search([True]),
                max_seq_len=10,
                attention_num_transformer_units=1,
                attention_num_heads=1,
                attention_head_dim=64,
                attention_position_wise_mlp_dim=32,
                # custom_model='saco_model',
                # custom_model='fcn_model',
                custom_model_config=dict(
                    env_cls=env_cls,
                    # 'embedding_hiddens': [64, 64],
                    # 
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
                    critic_concat_svo=False,
                    critic_concat_obs=True, # ──╮ ⛓ better lock together?
                    svo_concat_obs=True,    # ──╯ 
                    # onehot_attention=tune.grid_search([True, False]),
                    # svo_initializer='ortho',
                ),
                free_log_std=False,
            )
        )
        .rl_module(_enable_rl_module_api=False)
        .environment(env=env, render_env=False, env_config=env_config, disable_env_checking=False)
        .update_from_dict(dict(
            # == SaCo ==
            use_sa_and_svo=False, # whether use attention backbone or mlp backbone 
            use_fixed_svo=tune.grid_search([False]),
            fixed_svo=math.pi/4, #tune.grid_search([math.pi/4, math.pi/6, math.pi/3]),
            use_social_attention=True, # TODO
            use_svo=True, #tune.grid_search([True, False]), # whether or not to use svo to change reward, if False, use original reward
            # svo_init_value=tune.grid_search(['0', 'pi/4']),
            svo_mode='full', #tune.grid_search(['full', 'restrict']),
            nei_rewards_mode='mean_nei_rewards', #'attentive_one_nei_reward',
            sp_select_mode='numerical', # only work if use onehot_attention!
            # sp_select_mode=tune.grid_search(['bincount', 'numerical']), # only work if use onehot_attention!
            # nei_rewards_mode=tune.grid_search([
            #     'mean_nei_rewards',           # ─╮ 
            #     'max_nei_rewards',            #  │
            #     'nearest_nei_reward',         #  │──> Choose 1 alternatively
            #     'attentive_one_nei_reward',   #  │
            #     # 'attentive_all_nei_reward', # ─╯
            # ]),
            norm_adv=True,
            # == Common ==
            old_value_loss=False,
            # old_value_loss=True,
            num_neighbours=NUM_NEIGHBOURS,
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
