from ray.rllib.algorithms.algorithm import Algorithm

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from env.env_wrappers import get_rllib_cc_env, get_rllib_compatible_ma_env, get_neighbour_md_env
from env.env_utils import get_metadrive_ma_env_cls


TEST = False # <~~ Toggle TEST mod here! 
SCENE = "intersection" if not TEST else "intersection" 



if __name__ == "__main__":
    env_name, env_cls = get_rllib_compatible_ma_env(
                            get_neighbour_md_env(
                            get_metadrive_ma_env_cls(SCENE)), 
                            return_class=True)

    checkpoint_path = 'exp_SoCO/SOCO_Inter_30agents_v1(-a)(mean_nei_r)(svo_init_pi_6)(svo_coeff_1e-2)/SOCOTrainer_MultiAgentIntersectionEnv_6e8c2_00000_0_num_agents=30,start_seed=5000,nei_rewards_mode=mean_nei_rewards,svo_asymmetry__2023-07-28_17-16-27/checkpoint_001465'

    algo = Algorithm.from_checkpoint(checkpoint_path)
    print(algo)

    '''

    # === Environment ===
    env_name, env_cls = get_rllib_compatible_ma_env(
                            get_neighbour_md_env(
                            get_metadrive_ma_env_cls(SCENE)), 
                            return_class=True)

    # === Environmental Setting ===
    env_config = dict(
        num_agents=tune.grid_search(NUM_AGENTS),
        return_single_space=True,
        start_seed=tune.grid_search(seeds),
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
        neighbours_distance=10,
    )

    # if TEST
    stop, exp_name, num_rollout_workers = get_args_only_if_test(
                                            algo_name=ALGO_NAME, 
                                            env_config=env_config, 
                                            exp_des=EXP_DES, 
                                            scene=SCENE, 
                                            num_agents=NUM_AGENTS, 
                                            test=TEST) # env_config will be modified
    
    stop = {"timesteps_total": 1.5e6}
    if args.num_workers:
        num_rollout_workers = args.num_workers
    if TEST:
        # stop = {"timesteps_total": 3e6}
        stop = {"training_iteration": 10}
    if TEST and not args.num_agents:
        env_config['num_agents'] = 30


    # === Algo Setting ===
    algo_config = (
        SOCOConfig()
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
            )
        )
        .rl_module(_enable_rl_module_api=False)
        .environment(env=env_name, render_env=False, env_config=env_config, disable_env_checking=False)
        .update_from_dict(dict(
            # == SaCo ==
            test_new_rewards=tune.grid_search([True]),
            add_svo_loss=True,
            svo_loss_coeff=tune.grid_search([1]),
            svo_asymmetry_loss=tune.grid_search([False]),
            use_sa_and_svo=False, # whether use attention backbone or mlp backbone 
            use_fixed_svo=False,
            fixed_svo=math.pi/4, #tune.grid_search([math.pi/4, math.pi/6, math.pi/3]),
            use_social_attention=True, # TODO
            use_svo=True, #tune.grid_search([True, False]), # whether or not to use svo to change reward, if False, use original reward
            svo_init_value=tune.grid_search(['pi/6']), # in [ '0', 'pi/4', 'pi/2', 'random' ]
            svo_mode='full', #tune.grid_search(['full', 'restrict']),
            nei_rewards_mode=tune.grid_search(['mean_nei_rewards']), #'attentive_one_nei_reward',
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
            num_neighbours=NUM_NEIGHBOURS,
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

    '''