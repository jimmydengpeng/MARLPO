from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_trainable, register_env
    
from metadrive import (
    MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv, MultiAgentTollgateEnv,
    MultiAgentBottleneckEnv, MultiAgentParkingLotEnv
)

from marlpo.algo_ippo import IPPOConfig, IPPOTrainer
from marlpo.algo_arippo import ARIPPOConfig, ARIPPOTrainer
from marlpo.callbacks import MultiAgentDrivingCallbacks
from marlpo.env.env_wrappers import get_rllib_compatible_new_gymnasium_api_env
from marlpo.utils.utils import print, inspect, get_other_training_resources



# === Set a lot of configs ===

SCENE = "roundabout"
SEED = 5000
NUM_AGENTS = 4


ALL_CKP = dict(
    IPPO="exp_results/IPPO_CC_Roundabout_seed=5000_4agents/IPPOTrainer_MultiAgentRoundaboutEnv_b9bb5_00000_0_start_seed=5000_2023-05-18_20-22-58/checkpoint_000977",
    ARPPO="/Users/jimmy/ray_results/ARIPPO_V0_Roundabout_1seeds_NumAgentsSearch_4agents/ARIPPOTrainer_MultiAgentRoundaboutEnv_b0e1f_00000_0_start_seed=5000_2023-05-18_14-10-30/checkpoint_000977",
)
ckp = 'ARPPO'
CKP_DIR = ALL_CKP[ckp]

if ckp == 'IPPO':
    AlgoConfig = IPPOConfig
    AlgoTrainer = IPPOTrainer

elif ckp == 'ARPPO':
    AlgoConfig = ARIPPOConfig
    AlgoTrainer = ARIPPOTrainer
    MODEL_CONFIG={"custom_model_config": {'n_actions': NUM_AGENTS-1}}

def compute_actions_for_multi_agents(algo, obs):
    actions = {}
    for agent_id in obs:
        o = obs[agent_id]
        actions[agent_id] = algo.compute_single_action(o)
    return actions


if __name__ == "__main__":

    scenes = {
        "roundabout": MultiAgentRoundaboutEnv,
        "intersection": MultiAgentIntersectionEnv,
        "tollgate": MultiAgentTollgateEnv,
        "bottleneck": MultiAgentBottleneckEnv,
        "parkinglot": MultiAgentParkingLotEnv,
    }

    env, env_cls = get_rllib_compatible_new_gymnasium_api_env(scenes[SCENE], return_class=True)

   # === Environmental Setting ===
    env_config = dict(
        use_render=False,
        num_agents=NUM_AGENTS,
        return_single_space=True,
        # "manual_control": True,
        # crash_done=True,
        # "agent_policy": ManualControllableIDMPolicy
        start_seed=SEED
    )
    

    # === Algo Setting ===

    ppo_config = (
        AlgoConfig()
        .framework('torch')
        .resources(
            **get_other_training_resources()
        )
        .rollouts(
            num_rollout_workers=1,
        )
        .callbacks(MultiAgentDrivingCallbacks)
        .training(
            train_batch_size=1024,
            gamma=0.99,
            lr=3e-4,
            sgd_minibatch_size=512,
            num_sgd_iter=5,
            lambda_=0.95,
            model=MODEL_CONFIG,
        )
        .multi_agent(
        )
        # .evaluation(
        #     evaluation_interval=2,
        #     evaluation_duration=40,
        #     evaluation_config=dict(env_config=dict(environment_num=200, start_seed=0)),
        #     evaluation_num_workers=1,)
        .environment(env=env, render_env=False, env_config=env_config, disable_env_checking=False)
    )

    algo = AlgoTrainer(config=ppo_config)
    algo.load_checkpoint(CKP_DIR)
    # env = algo.workers.local_worker().env

    env = env_cls(env_config)
    obs, info = env.reset()

    stop_render = False
    max_render_epi = 100
    cur_epi = 0
    while not stop_render:
        
        actions = compute_actions_for_multi_agents(algo, obs)
        obs, r, term, trunc, info = env.step(actions)

        if term['__all__']:
            cur_epi += 1
            print(f'episode {cur_epi} ends.')
            obs, info = env.reset()
            if cur_epi > max_render_epi:
                stop_render = True

        env.render(mode="top_down", film_size=(1000, 1000)) 

    env.close()



# sample_batch = worker.sample()
# print(sample_batch)
# policy = worker.get_policy()
# print(policy)
# worker.preprocessors['default_policy'] = lambda x: x
# print(worker.preprocessors)
# preprocessed = worker.preprocessors[policy_id].transform(ob)
# env = MultiAgentMetaDriveEnv(config)
# obs, info = env.reset()
# print(type(obs))
# for agent_id, ob in obs.items():
#     print(agent_id, ob)
# a = ppo_algo.compute_actions(obs)

