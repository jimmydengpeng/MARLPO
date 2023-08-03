import math
from ray import tune

from algo import SOCOConfig, SOCOTrainer
from callbacks import MultiAgentDrivingCallbacks
from env.env_wrappers import get_rllib_compatible_env, get_neighbour_md_env
from env.env_utils import get_metadrive_ma_env_cls
from train import train
from utils import (
    get_train_parser, 
    get_training_resources, 
    get_other_training_configs,
)


TEST = False # <~~ Toggle TEST mod here! 
# TEST = True

ALGO_NAME = "SOCO"
SCENE = "intersection" if not TEST else "intersection" 

# === Env Seeds ===
# seeds = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
# seeds = [5000, 6000, 7000]
SEEDS = [6000]
# seeds = [8000, 9000, 10000, 11000, 12000]

# NUM_AGENTS = [30]
NUM_AGENTS = [30]
NUM_NEIGHBOURS = 8
EXP_DES = "v2[mean_nei_r][max_local_r][nei_mean_r_coeff=0.5]"

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    
    num_agents, exp_name, num_rollout_workers, SEEDS, TEST = get_other_training_configs(
                                                                args=args,
                                                                algo_name=ALGO_NAME, 
                                                                exp_des=EXP_DES, 
                                                                scene=SCENE, 
                                                                num_agents=NUM_AGENTS,
                                                                seeds=SEEDS,
                                                                test=TEST) 
    # == Get Environment ==
    env_name, env_cls = get_rllib_compatible_env(
                            get_neighbour_md_env(
                            get_metadrive_ma_env_cls(SCENE)), 
                            return_class=True)

    # == Environmental Setting ==
    env_config = dict(
        num_agents=tune.grid_search(num_agents),
        return_single_space=True,
        start_seed=tune.grid_search(SEEDS),
        delay_done=25,
        vehicle_config=dict(
            lidar=dict(
                num_lasers=72, 
                distance=40, 
                num_others=0,
            )
        ),
        # == neighbour config ==
        use_dict_obs=False,
        add_compact_state=False, # add BOTH ego- & nei- compact-state simultaneously
        add_nei_state=False,
        num_neighbours=NUM_NEIGHBOURS, # determine how many neighbours's abs state will be included in obs
        neighbours_distance=tune.grid_search([15]),
    )

    # ========== changable =========== 
    stop = {"timesteps_total": 1.2e6}
    if TEST:
        stop = {"training_iteration": 10}
        # env_config['num_agents'] = 30
    # ================================ 


    # === Algo Setting ===
    algo_config = (
        SOCOConfig()
        .framework('torch')
        .resources(
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
            lr=3e-4,
            sgd_minibatch_size=512,
            num_sgd_iter=5,
            lambda_=0.95,
            model=dict(
                use_attention=False,
                max_seq_len=10,
                attention_num_transformer_units=1,
                attention_num_heads=2,
                attention_head_dim=32,
                attention_position_wise_mlp_dim=32,
                # custom_model='saco_model',
                # custom_model='fcn_model',
                custom_model_config=dict(
                    # == fcn model config ==
                    uniform_init_svo=False,

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
            ),
            _enable_learner_api=False,
        )
        .rl_module(_enable_rl_module_api=False)
        .environment(env=env_name, render_env=False, env_config=env_config, disable_env_checking=True)
        .update_from_dict(dict(
            # == SaCo ==
            test_new_rewards=False, # DEPRECATED! not in use! refer to 'reward_coor_mode'
            nei_reward_if_no_nei='self', # ['0']
            add_svo_loss=False,
            svo_loss_coeff=tune.grid_search([1]),
            svo_asymmetry_loss=False,
            use_sa_and_svo=False, # whether use attention backbone or mlp backbone 
            use_fixed_svo=False,
            fixed_svo=math.pi/8, #tune.grid_search([math.pi/4, math.pi/6, math.pi/3]),
            use_social_attention=True, # TODO
            use_svo=True, #tune.grid_search([True, False]), # whether or not to use svo to change reward, if False, use original reward
            svo_init_value='0', # in [ '0', 'pi/4', 'pi/2', 'random' ]
            svo_mode='full', #tune.grid_search(['full', 'restrict']),
            nei_rewards_mode=tune.grid_search(['mean_nei_rewards']), 
            # == add: r = ego_r + nei_r ==
            # == local_mean: r = mean(ego_r + num_nei * nei_r) ==
            reward_coor_mode=tune.grid_search(['add']), # default: None=='ego', 'svo', 'add'
            nei_r_add_coeff=tune.grid_search([0.1, 0.5]),
            
            sp_select_mode='numerical', # only work if use onehot_attention!
            # sp_select_mode=tune.grid_search(['bincount', 'numerical']), # only work if use onehot_attention!
            # nei_rewards_mode=tune.grid_search([
            #     'mean_nei_rewards',           # ─╮ 
            #     'max_nei_rewards',            #  │
            #     'nearest_nei_reward',         #  │──> Choose 1 alternatively
            #     'attentive_one_nei_reward',   #  │
            #     # 'attentive_all_nei_reward', # ─╯
            # ]),
            norm_adv=tune.grid_search([True]),
            # == Common ==
            old_value_loss=False,
            # old_value_loss=True,
            num_neighbours=tune.grid_search([NUM_NEIGHBOURS]),
            # == CC ==
            use_central_critic=False,
            counterfactual=False,
            fuse_mode="none",
        ))
    )

    # === Launch training ===
    train(
        SOCOTrainer,
        config=algo_config,
        stop=stop,
        exp_name=exp_name,
        checkpoint_freq=10,
        keep_checkpoints_num=3,
        num_gpus=0,
        results_path='exp_SoCO',
        test_mode=TEST,
    )
