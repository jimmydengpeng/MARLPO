from collections import defaultdict

import numpy as np

from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_trainable, register_env
    
from metadrive import (
    MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv, MultiAgentTollgateEnv,
    MultiAgentBottleneckEnv, MultiAgentParkingLotEnv
)

from marlpo.algo.algo_ippo import IPPOConfig, IPPOTrainer
from marlpo.algo_arippo import ARIPPOConfig, ARIPPOTrainer
from marlpo.algo_ccppo import CCPPOConfig, CCPPOTrainer
from marlpo.algo_arccppo import ARCCPPOConfig, ARCCPPOTrainer
from marlpo.callbacks import MultiAgentDrivingCallbacks
from marlpo.env.env_wrappers import get_rllib_compatible_gymnasium_api_env, get_ccppo_env
from marlpo.utils.debug import print, inspect
from marlpo.utils.utils import get_other_training_resources, get_num_workers



# === Set a lot of configs ===

# SCENE = "roundabout"
SCENE = "intersection"
ALGO = 'IPPO'
# ALGO = 'ARPPO'
# ALGO = 'CCPPO'
# ALGO = 'ARCCPPO'
FUSE_MODE = 'concat' 
RANDOM_ORDER = True
# FUSE_MODE = 'mf'

SEED = 5000
NUM_AGENTS = 30

EVAL_ENV_NUM_AGENTS = NUM_AGENTS
EVAL_ENV_NUM_AGENTS = 30

ALL_CKP = dict(
    # IPPO_4a_5000="exp_results/IPPO_CC_Roundabout_seed=5000_4agents/IPPOTrainer_MultiAgentRoundaboutEnv_b9bb5_00000_0_start_seed=5000_2023-05-18_20-22-58/checkpoint_000977",
    IPPO_4a_5000="/Users/jimmy/ray_results/IPPO_Central_Value_Roundabout_8seeds/IPPOTrainer_MultiAgentRoundaboutEnv_ac78b_00000_0_start_seed=5000_2023-04-20_18-04-01/checkpoint_000977",
    IPPO_40a_5000_intersection='exp_results/IPPO_Intersection_seed=5000_40agents/IPPOTrainer_MultiAgentIntersectionEnv_437fb_00000_0_start_seed=5000_2023-06-01_17-39-54/checkpoint_000977',
    IPPO_30a_5000_intersection='exp_results/IPPO_Intersection_8seeds_30agents/IPPOTrainer_MultiAgentIntersectionEnv_c9a21_00000_0_start_seed=5000_seed=0_2023-06-06_20-08-03/checkpoint_000977',
    ARPPO_4a_5000="/Users/jimmy/ray_results/ARIPPO_V0_Roundabout_1seeds_NumAgentsSearch_4agents/ARIPPOTrainer_MultiAgentRoundaboutEnv_b0e1f_00000_0_start_seed=5000_2023-05-18_14-10-30/checkpoint_000977",
    ARPPO_40a_5000="/Users/jimmy/ray_results/ARIPPO_V0_Roundabout_8seeds/ARIPPOTrainer_MultiAgentRoundaboutEnv_cc1dc_00000_0_start_seed=5000_2023-05-17_23-02-09/checkpoint_000977",
    ARPPO_32a_5000="/Users/jimmy/ray_results/ARIPPO_V0_Roundabout_seed=5000_NumAgentsSearch_32agents/ARIPPOTrainer_MultiAgentRoundaboutEnv_11848_00000_0_start_seed=5000_2023-05-18_15-53-25/checkpoint_000977",
    ARPPO_16a_5000="/Users/jimmy/ray_results/ARIPPO_V0_Roundabout_seed5000seeds_NumAgentsSearch_16agents/ARIPPOTrainer_MultiAgentRoundaboutEnv_6f6c6_00000_0_start_seed=5000_2023-05-18_14-51-37/checkpoint_000977",
    ARPPO_8a_5000="/Users/jimmy/ray_results/ARIPPO_V0_Roundabout_1seeds_NumAgentsSearch_8agents/ARIPPOTrainer_MultiAgentRoundaboutEnv_c3401_00000_0_start_seed=5000_2023-05-18_14-25-20/checkpoint_000977",
    CCPPO_concat_4a_5000="exp_results/CCPPO_Roundabout_4seeds_4agents/CCPPOTrainer_MultiAgentRoundaboutEnv_c9184_00000_0_start_seed=5000_fuse_mode=concat_2023-05-21_19-58-22/checkpoint_000977",
    CCPPO_concat_40a_5000="/Users/jimmy/ray_results/CCPPO_Roundabout_8seeds/CCPPOTrainer_MultiAgentRoundaboutEnv_a9c37_00000_0_start_seed=5000_fuse_mode=concat_2023-04-21_13-02-06/checkpoint_000977",
    ARPPO_4a_5000_intersection="exp_results/ARIPPO_V0_Intersection_4seeds_4agents/ARIPPOTrainer_MultiAgentIntersectionEnv_31282_00000_0_start_seed=5000_2023-05-22_13-47-52/checkpoint_000977",
    ARPPO_4a_8000_intersection="exp_results/ARIPPO_V0_Intersection_4seeds_4agents/ARIPPOTrainer_MultiAgentIntersectionEnv_31282_00003_3_start_seed=8000_2023-05-22_14-46-31/checkpoint_000977",
    # ARPPO_4a_8000_intersection="exp_results/ARIPPO_V0_Intersection_seed=8000_4agentsmaxsteps=1e7/ARIPPOTrainer_MultiAgentIntersectionEnv_8df55_00000_0_start_seed=8000_2023-05-22_17-18-03/checkpoint_009766",
    # ARPPO_4a_8000_intersection="exp_results/ARIPPO_V0_Intersection_seed=8000_4agentsmaxsteps=1e7/ARIPPOTrainer_MultiAgentIntersectionEnv_8df55_00000_0_start_seed=8000_2023-05-22_17-18-03/checkpoint_001960",
    CCPPO_concat_4a_8000_intersection="exp_results/CCPPO_Intersection_4seeds_4agents/CCPPOTrainer_MultiAgentIntersectionEnv_c8dd5_00003_3_start_seed=8000_fuse_mode=concat_2023-05-22_16-23-18/checkpoint_000977",
    IPPO_4a_8000_intersection="exp_results/IPPO_Intersection_seed=8000_4agents/IPPOTrainer_MultiAgentIntersectionEnv_62b1e_00000_0_start_seed=8000_2023-05-23_17-44-17/checkpoint_000977",
    CCPPO_mf_4a_8000_intersection="exp_results/CCPPO_Intersection_seed=8000_4agents/CCPPOTrainer_MultiAgentIntersectionEnv_7b14d_00000_0_start_seed=8000_fuse_mode=mf_2023-05-23_22-09-50/checkpoint_000977",
    ARCCPPO_concat_4a_5000="exp_results/ARCCPPO_Roundabout_seed=5000_40agents_V0/ARCCPPOTrainer_MultiAgentRoundaboutEnv_51fd6_00001_1_start_seed=5000_fuse_mode=concat_2023-05-31_15-02-05/checkpoint_000840",
    ARCCPPO_concat_40a_5000="exp_results/ARCCPPO_Roundabout_seed=5000_40agents_V0/ARCCPPOTrainer_MultiAgentRoundaboutEnv_51fd6_00001_1_start_seed=5000_fuse_mode=concat_2023-05-31_15-02-05/checkpoint_000977",
    ARCCPPO_mf_40a_5000_intersection="exp_results/ARCCPPO_Intersection_seed=5000_40agents_V0/ARCCPPOTrainer_MultiAgentIntersectionEnv_3b3e0_00000_0_start_seed=5000_fuse_mode=mf_2023-05-31_12-04-25/checkpoint_000977",
    ARCCPPO_concat_40a_5000_intersection="exp_results/ARCCPPO_Intersection_seed=5000_40agents_V1/ARCCPPOTrainer_MultiAgentIntersectionEnv_e59bc_00002_2_counterfactual=True_start_seed=5000_fuse_mode=concat_2023-06-01_01-31-52/checkpoint_000977",
    ARCCPPO_concat_ro_40a_5000_intersection='exp_results/ARCCPPO_Roundabout_seed=5000_40agents_ro/ARCCPPOTrainer_MultiAgentRoundaboutEnv_eda47_00001_1_start_seed=5000_fuse_mode=concat_random_order=True_2023-06-02_21-55-34/checkpoint_000750'
)
# ckp = 'ARPPO_32a_5000'


if ALGO == 'IPPO':
    AlgoConfig = IPPOConfig
    AlgoTrainer = IPPOTrainer
    MODEL_CONFIG={}
    OTHER_CONFIG = {}

elif ALGO == 'ARPPO':
    AlgoConfig = ARIPPOConfig
    AlgoTrainer = ARIPPOTrainer
    MODEL_CONFIG={"custom_model_config": {'n_actions': NUM_AGENTS-1}}
    OTHER_CONFIG = {}
elif ALGO == 'CCPPO':
    AlgoTrainer = CCPPOTrainer
    AlgoConfig = CCPPOConfig
    MODEL_CONFIG = {}
    OTHER_CONFIG = dict(
        counterfactual=True,
        # mf_nei_distance=10,
        fuse_mode=FUSE_MODE,
    )
    ALGO = ALGO + '_' + FUSE_MODE
elif  ALGO == 'ARCCPPO':
    AlgoTrainer = ARCCPPOTrainer
    AlgoConfig = ARCCPPOConfig
    MODEL_CONFIG = {
        "custom_model_config": {
            "num_neighbours": 4,
        }
    }
    OTHER_CONFIG = dict(
        counterfactual=True,
        # mf_nei_distance=10,
        fuse_mode=FUSE_MODE,
    )
    ALGO = ALGO + '_' + FUSE_MODE
    if RANDOM_ORDER:
        ALGO += '_ro'
        OTHER_CONFIG.update(dict(
            random_order=RANDOM_ORDER
        ))

else:
    raise NotImplementedError

# === get ckp path ===
ckp = "_".join([ALGO, str(NUM_AGENTS)+'a', str(SEED)])
if SCENE == 'intersection':
    ckp = ckp + '_' + SCENE
CKP_DIR = ALL_CKP[ckp]


def compute_actions_for_multi_agents_in_batch(algo, obs, infos):
    info_list = []
    assert isinstance(infos, dict)
    for k in obs:
        infos[k]['agent_id'] = k
        info_list.append(infos[k])
        
    actions = algo.compute_actions(obs, info=info_list, explore=False)
    return actions

def compute_actions_for_multi_agents_separately(algo, obs):
    actions = {}
    for agent_id in obs:
        o = obs[agent_id]
        actions[agent_id] = algo.compute_single_action(o)
    return actions


# === Metrics callbacks ===
class MetricCallbacks():

    def __init__(self) -> None:
        pass

    def _setup(self):
        self._last_infos = {}
        self.data = {}

    def on_episode_start(self):
        self._setup()

        # data["velocity"] = defaultdict(list)
        # data["steering"] = defaultdict(list)
        # data["step_reward"] = defaultdict(list)
        # data["acceleration"] = defaultdict(list)
        # data["episode_length"] = defaultdict(list)
        # data["episode_reward"] = defaultdict(list)
        # data["num_neighbours"] = defaultdict(list)

    def on_episode_step(self, infos, active_keys=None):
        for agent_id, info in infos.items():
            self.set_last_info(agent_id, info)
        
        # for agent_id in active_keys:
        #     k = agent_id
        #     info = last_infos.get(k)
        #     if info:
        #         if "step_reward" not in info:
        #             continue
        #         data["velocity"][k].append(info["velocity"])
        #         data["steering"][k].append(info["steering"])
        #         data["step_reward"][k].append(info["step_reward"])
        #         data["acceleration"][k].append(info["acceleration"])
        #         data["episode_length"][k].append(info["episode_length"])
        #         data["episode_reward"][k].append(info["episode_reward"])
        #         data["num_neighbours"][k].append(len(info.get("neighbours", []))) # may not contain

    def on_episode_end(self, metrics):
        arrive_dest_list = []
        crash_list = []
        out_of_road_list = []
        max_step_list = []
        epi_rew_list = []
        epi_len_list = []
        for k in self._last_infos:
            info = self._last_infos[k]

            arrive_dest = info.get("arrive_dest", False)
            crash = info.get("crash", False)
            out_of_road = info.get("out_of_road", False)
            max_step = not (arrive_dest or crash or out_of_road)

            arrive_dest_list.append(arrive_dest)
            crash_list.append(crash)
            out_of_road_list.append(out_of_road)
            max_step_list.append(max_step)

            epi_rew_list.append(info.get("episode_reward", 0))
            epi_len_list.append(info.get("episode_length", 0))

        # === 计算 成功率、撞车、出界率、最大步数率 ===
        metrics["success_rate"] = np.mean(arrive_dest_list)
        metrics["crash_rate"] = np.mean(crash_list)
        metrics["out_of_road_rate"] = np.mean(out_of_road_list)
        metrics["max_step_rate"] = np.mean(max_step_list)

        # === 计算 平均奖励、平均长度 ===
        metrics["epsode_reward_mean"] = np.mean(epi_rew_list)
        metrics["epsode_length_mean"] = np.mean(epi_len_list)

    def set_last_info(self, agent_id, info):
        self._last_infos[agent_id] = info


def print_final_summary(rate_lists):
    # epi_succ_rate_list, epi_crash_rate_list, epi_out_rate_list, epi_max_step_rate_list
    rates = []
    for rate_list in rate_lists:
        rates.append(np.mean(rate_list).round(2))

    # res = []
    prefix_strs = ('succ_rate', 'crash_rate', 'out_rate', 'maxstep_rate')
    for prefix, rate in zip(prefix_strs, rates):
        res = prefix+': '+str(rate*100)+'%'
        print(res, end='\n')
    

if __name__ == "__main__":

    scenes = {
        "roundabout": MultiAgentRoundaboutEnv,
        "intersection": MultiAgentIntersectionEnv,
        "tollgate": MultiAgentTollgateEnv,
        "bottleneck": MultiAgentBottleneckEnv,
        "parkinglot": MultiAgentParkingLotEnv,
    }

    # if 'ARCCPPO' in ALGO:
    env, env_cls = get_ccppo_env(scenes[SCENE], return_class=True)
    # else:
    #     env, env_cls = get_rllib_compatible_gymnasium_api_env(scenes[SCENE], return_class=True)

   # === Environmental Setting ===
    env_config = dict(
        use_render=False,
        num_agents=EVAL_ENV_NUM_AGENTS,
        return_single_space=True,
        # "manual_control": True,
        # crash_done=True,
        # "agent_policy": ManualControllableIDMPolicy
        # delay_done=0,
        start_seed=SEED
    )
    

    # === Algo Setting ===

    algo_config = (
        AlgoConfig()
        .debugging(seed=0)
        .framework('torch')
        .resources(
            **get_other_training_resources()
        )
        .rollouts(
            num_rollout_workers=0,
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
        .environment(env=env, render_env=False, env_config=env_config, disable_env_checking=False)
        .update_from_dict(OTHER_CONFIG)
    )

    # from ray.rllib.algorithms.algorithm import Algorithm
    # algo = Algorithm.from_checkpoint(checkpoint_path)

    algo = AlgoTrainer(config=algo_config)
    # algo = AlgoTrainer(config=ALGO_CONFIG)
    algo.load_checkpoint(CKP_DIR)
    # inspect(algo.workers)
    # inspect(algo.workers.local_worker())
    # inspect(algo.workers.local_worker().preprocessors)
    # env = algo.workers.local_worker().env

    # === init metric callbacks ===
    callbacks = MetricCallbacks()

    env = env_cls(env_config)
    obs, infos = env.reset()
    callbacks.on_episode_start()

    stop_render = False
    NUM_EPISODES_TOTAL = 10
    cur_epi = 0

    # RENDER = False
    RENDER = True

    episodic_mean_rews = []
    episodic_mean_succ_rate = []
    episodic_mean_out_rate = []
    episodic_mean_crash_rate = []

    epi_mean_rews = 0
    epi_mean_succ_rate = 0
    epi_mean_out_rate = 0
    epi_mean_crash_rate = 0


    last_infos = {}

    data = {
        "episode_reward": defaultdict(list)
    }
    metrics = {
        "success_rate": np.float32(0),
        "crash_rate": np.float32(0),
        "out_of_road_rate": np.float32(0),
        "max_step_rate": np.float32(0),
        "epsode_length_mean": np.float32(0),
        "epsode_reward_mean": np.float32(0),
    }

    epi_succ_rate_list = []
    epi_crash_rate_list = []
    epi_out_rate_list = []
    epi_max_step_rate_list = []

    while not stop_render:
        
        # if ALGO == 'ARPPO' or ALGO == 'ARCCPPO_concat':
        actions = compute_actions_for_multi_agents_in_batch(algo, obs, infos)
        # else:
        #     actions = compute_actions_for_multi_agents_separately(algo, obs, infos)

        obs, rew, term, trunc, infos = env.step(actions)
        callbacks.on_episode_step(infos=infos)

        # on_episode_step()

            

        # === episode end ===
        if term['__all__']:
            cur_epi += 1
            callbacks.on_episode_end(metrics)
            epi_succ_rate_list.append(metrics["success_rate"])
            epi_crash_rate_list.append(metrics["crash_rate"])
            epi_out_rate_list.append(metrics["out_of_road_rate"])
            epi_max_step_rate_list.append(metrics["max_step_rate"])
            print(f'episode #{cur_epi} ends.')
            print(metrics)

            if cur_epi >= NUM_EPISODES_TOTAL:
                stop_render = True
                break

            callbacks.on_episode_start()
            obs, infos = env.reset()
           
        if RENDER:
            env.render(mode="top_down", film_size=(1000, 1000))

    env.close()
    print_final_summary((
        epi_succ_rate_list, 
        epi_crash_rate_list, 
        epi_out_rate_list, 
        epi_max_step_rate_list
    ))



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

