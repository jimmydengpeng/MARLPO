from collections import defaultdict
from typing import Dict

import numpy as np

from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_trainable, register_env
    
from metadrive import (
    MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv, MultiAgentTollgateEnv,
    MultiAgentBottleneckEnv, MultiAgentParkingLotEnv
)

# from marlpo.algo.algo_ippo import IPPOConfig, IPPOTrainer
# from marlpo.algo_arippo import ARIPPOConfig, ARIPPOTrainer
# from marlpo.algo_ccppo import CCPPOConfig, CCPPOTrainer
# from marlpo.algo_arccppo import ARCCPPOConfig, ARCCPPOTrainer
# from marlpo.algo_sappo import SAPPOConfig, SAPPOTrainer
# from marlpo.callbacks import MultiAgentDrivingCallbacks
# from marlpo.env.env_wrappers import get_rllib_compatible_gymnasium_api_env, get_ccppo_env
from utils.debug import print, printPanel

from env.env_wrappers import get_rllib_compatible_env, get_tracking_md_env

# === Set a lot of configs ===

# SCENE = "roundabout"
SCENE = "intersection"
# ALGO = 'IPPO'
# ALGO = 'ARPPO'
# ALGO = 'CCPPO'
# ALGO = 'ARCCPPO'
ALGO = 'SAPPO'
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
    ARCCPPO_concat_ro_40a_5000_intersection='exp_results/ARCCPPO_Roundabout_seed=5000_40agents_ro/ARCCPPOTrainer_MultiAgentRoundaboutEnv_eda47_00001_1_start_seed=5000_fuse_mode=concat_random_order=True_2023-06-02_21-55-34/checkpoint_000750',

    # SAPPO_30a_5000_intersection='/Users/jimmy/Projects/RL/My_RLlib_Algo/exp_results/SAPPO_Inter_4-8-16-30agents_v5(svo)_use_critic_loss_update_svo/SAPPOTrainer_MultiAgentIntersectionEnv_eec86_00001_1_num_agents=8_start_seed=5000_2023-07-14_23-59-12/checkpoint_002330',
    SAPPO_30a_5000_intersection='/Users/jimmy/Projects/RL/My_RLlib_Algo/exp_results/SAPPO_Inter_4-8-16-30agents_v5(svo)_use_critic_loss_update_svo/SAPPOTrainer_MultiAgentIntersectionEnv_eec86_00003_3_num_agents=30_start_seed=5000_2023-07-15_10-20-23/checkpoint_002930',
    SAPPO_8a_5000_intersection='/Users/jimmy/Projects/RL/My_RLlib_Algo/exp_results/SAPPO_Inter_4-8-16-30agents_v5(svo)_use_critic_loss_update_svo/SAPPOTrainer_MultiAgentIntersectionEnv_eec86_00001_1_num_agents=8_start_seed=5000_2023-07-14_23-59-12/checkpoint_002330',
    SAPPO_16a_5000_intersection='/Users/jimmy/Projects/RL/My_RLlib_Algo/exp_results/SAPPO_Inter_4-8-16-30agents_v5(svo)_use_critic_loss_update_svo/SAPPOTrainer_MultiAgentIntersectionEnv_eec86_00002_2_num_agents=16_start_seed=5000_2023-07-15_03-24-06/checkpoint_000810',
)
# ckp = 'ARPPO_32a_5000'


# === Metrics callbacks ===
class MetricCallbacks():

    def __init__(self) -> None:
        pass

    def _setup(self):
        self._last_infos = {}
        self.data = {}
        self.env_steps = 0

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
        self.env_steps += 1
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
        metrics["num_success"] = np.count_nonzero(arrive_dest_list)
        metrics["total_agents"] = len(arrive_dest_list)
        metrics["crash_rate"] = np.mean(crash_list)
        metrics["out_of_road_rate"] = np.mean(out_of_road_list)
        metrics["max_step_rate"] = np.mean(max_step_list)

        # === 计算 平均奖励、平均长度 ===
        metrics["epsode_reward_mean"] = np.mean(epi_rew_list)
        metrics["epsode_length_mean"] = np.mean(epi_len_list)

        metrics["env_steps"] = self.env_steps

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
    



def test():
    from ray.rllib.algorithms.algorithm import Algorithm

    checkpoint_path = 'exp_results/SACO_Inter_8agents_(saco)/SCPPOTrainer_MultiAgentIntersectionEnv_80870_00001_1_num_agents=8_start_seed=5000_sp_select_mode=numerical_2023-07-20_15-17-42/checkpoint_000550'
    algo = Algorithm.from_checkpoint(checkpoint_path)
    print('>>> ', algo.config['sp_select_mode']) 
    algo.compute_single_action()


def get_algo_new():
    from ray.rllib.algorithms.algorithm import Algorithm

    ScCO_16a_5000_intersection_4others_nei_dis_20m='exp_SoCO/SOCO_Inter_16agents_v0/SOCOTrainer_MultiAgentIntersectionEnv_a5911_00000_0_num_agents=16,start_seed=5000,num_others=4,use_attention=False,use_fixed_svo=F_2023-07-26_20-55-07/checkpoint_000977'
    SoCO_30a_5000_intersection='exp_results/SACO_Inter_30agents_(saco)/SCPPOTrainer_MultiAgentIntersectionEnv_e73a0_00000_0_num_agents=30_start_seed=5000_use_fixed_svo=False_2023-07-23_22-47-53/checkpoint_000940'

    SoCO_30a_5000_intersection_atn='exp_results/SACO_Inter_30agents_(saco)/SCPPOTrainer_MultiAgentIntersectionEnv_aac0d_00000_0_num_agents=30_start_seed=5000_use_attention=True_use_fixed_svo=False_2023-07-24_22-45-00/checkpoint_000977'

    SoCO_30a_5000_intersection_add_others='exp_results/SACO_Inter_30agents_(saco)/SCPPOTrainer_MultiAgentIntersectionEnv_c6747_00000_0_num_agents=30,start_seed=5000,num_others=4,use_attention=False,use_fixed_svo=_2023-07-26_12-06-19/checkpoint_000977'

    ScCO_30a_5000_intersection_no_others_nei_dis_10m='exp_SoCO/SOCO_Inter_30agents_v1/SOCOTrainer_MultiAgentIntersectionEnv_4f5b8_00000_0_neighbours_distance=10,num_agents=30,start_seed=5000,num_others=0,use_attentio_2023-07-26_22-47-14/checkpoint_000977' # BEST
    ScCO_30a_5000_intersection_no_others_nei_dis_10m_max_r_init_0='exp_SoCO/SOCO_Inter_30agents_v1_-a_max_nei_r_init_0/SOCOTrainer_MultiAgentIntersectionEnv_1c238_00000_0_neighbours_distance=10,num_agents=30,start_seed=5000,num_others=0,nei_rewards__2023-07-27_22-30-18/checkpoint_000977'
    ScCO_30a_5000_intersection_no_others_nei_dis_20m='exp_SoCO/SOCO_Inter_30agents_v1/SOCOTrainer_MultiAgentIntersectionEnv_4f5b8_00001_1_neighbours_distance=20,num_agents=30,start_seed=5000,num_others=0,use_attentio_2023-07-26_22-47-14/checkpoint_000977'

    ScCO_30a_5000_intersection_no_others_nei_dis_40m='exp_SoCO/SOCO_Inter_30agents_v1/SOCOTrainer_MultiAgentIntersectionEnv_4f5b8_00002_2_neighbours_distance=40,num_agents=30,start_seed=5000,num_others=0,use_attentio_2023-07-26_22-47-14/checkpoint_000977'

    ScCO_30a_5000_intersection_no_others_nei_dis_10m_coeff_1='exp_SoCO/SOCO_Inter_30agents_v1(-a)(mean_nei_r)(svo_init_pi_6)(svo_coeff_1e-1)/SOCOTrainer_MultiAgentIntersectionEnv_55553_00000_0_num_agents=30,start_seed=5000,nei_rewards_mode=mean_nei_rewards,svo_asymmetry__2023-07-28_17-15-45/checkpoint_001460'

    SoCO_DC='exp_SoCO/SOCO_DC_Inter_30agents_v0(-a)(mean_nei_r)(svo_init_pi_6)(svo_coeff_1)/SOCODCTrainer_MultiAgentIntersectionEnv_95369_00002_2_num_agents=30,start_seed=5000,norm_adv=True,svo_init_value=pi_6,svo_loss_coe_2023-07-31_02-12-02/checkpoint_000977'
    SoCO_DC_2='exp_SoCO/SOCO_DC_Inter_30agents_v0(-a)(mean_nei_r)(svo_init_pi_6)(svo_coeff_1)/SOCODCTrainer_MultiAgentIntersectionEnv_95369_00001_1_num_agents=30,start_seed=5000,norm_adv=False,svo_init_value=pi_6,svo_loss_co_2023-07-31_02-12-02/checkpoint_000977'

    ippo='exp_results/IPPO_Intersection_8seeds_30agents_repeat/IPPOTrainer_MultiAgentIntersectionEnv_3f8db_00000_0_start_seed=5000_seed=0_2023-06-07_19-05-44/checkpoint_000977'

    ippo_16a_4others='exp_SoCO/IPPO_Inter_16agents_(obs=107)/IPPOTrainer_MultiAgentIntersectionEnv_d0a96_00000_0_num_agents=16,start_seed=5000,num_others=4_2023-07-26_20-49-10/checkpoint_000977'

    ippo_30a_obs_91='exp_SoCO/IPPO_Inter_30agents_(obs=91)/IPPOTrainer_MultiAgentIntersectionEnv_b73e8_00000_0_num_agents=30,start_seed=5000,num_others=0_2023-07-27_13-09-08/checkpoint_000977'

    ippo_30a_obs_91_2='exp_SoCO/IPPO_Inter_30agents_(obs=91)/IPPOTrainer_MultiAgentIntersectionEnv_325a1_00000_0_num_agents=30,start_seed=6000,num_others=0_2023-07-28_18-12-02/checkpoint_000977'

    ippo_5000='exp_SoCO/IPPO_Inter_4agents_(8_seeds)(4_workers)/IPPOTrainer_MultiAgentIntersectionEnv_06981_00000_0_num_agents=30,start_seed=5000,num_others=0_2023-08-01_21-12-11/checkpoint_001172'

    pth='exp_SoCO/SOCO_Inter_30agents_v1(-a)(mean_nei_r)(svo_init_pi_6)(svo_coeff_1e-2)/SOCOTrainer_MultiAgentIntersectionEnv_6e8c2_00000_0_num_agents=30,start_seed=5000,nei_rewards_mode=mean_nei_rewards,svo_asymmetry__2023-07-28_17-16-27/checkpoint_001465'

    ippo_best='exp_SoCO/IPPO_Inter_30agents_(obs=91)/IPPOTrainer_MultiAgentIntersectionEnv_e8564_00000_0_num_agents=30,start_seed=5000,num_others=0_2023-07-29_22-05-00/checkpoint_001730'

    max_ego_and_nei='exp_SoCO/SOCO_Inter_30agents_v2[mean_nei_r][max_local_r])/SOCOTrainer_MultiAgentIntersectionEnv_86376_00001_1_num_agents=30,start_seed=5000,nei_rewards_mode=mean_nei_rewards,norm_adv=False_2023-08-01_22-41-39/checkpoint_000610'
    max_ego_and_nei_norm_a='exp_SoCO/SOCO_Inter_30agents_v2[mean_nei_r][max_local_r])/SOCOTrainer_MultiAgentIntersectionEnv_86376_00000_0_num_agents=30,start_seed=5000,nei_rewards_mode=mean_nei_rewards,norm_adv=True,_2023-08-01_22-41-39/checkpoint_001090'

    ppp='exp_IRAT/IRAT_Inter_30agents_v5/IRATTrainer_Intersection_8c77d_00000_0_num_agents=30,start_seed=5000,nei_rewards_mode=sum_2023-08-06_19-15-28/checkpoint_000830'
    tmp='exp_IRAT/IRAT_Inter_30agents_v5-self_1_neir_1/IRATTrainer_Intersection_3fd5f_00000_0_neighbours_distance=15,num_agents=30,start_seed=5000,nei_rewards_add_coeff=1,nei_rewards_mo_2023-08-08_14-10-18/checkpoint_000820'

    irat_np_clip='exp_IRAT/IRAT_Inter_30agents_v5-more-alt-1-no-grad-clip/IRATTrainer_Intersection_54688_00000_0_neighbours_distance=40,num_agents=30,start_seed=5000,nei_rewards_add_coeff=1,nei_rewards_mo_2023-08-09_20-14-46/checkpoint_000977'

    ippo_='exp_baselines/IPPO_Inter_4agents_(8_seeds)(4_workers)/IPPOTrainer_MultiAgentIntersectionEnv_7d9c8_00000_0_num_agents=30,start_seed=5000,num_others=0_2023-08-02_15-16-25/checkpoint_000920'

    ippo_tmp='exp_IPPO/IPPO_Inter_4agents_(7_workers)(no_norm_adv)/IPPOTrainer_Intersection_2dba9_00000_0_num_agents=30,start_seed=5000,num_others=0_2023-08-10_01-50-07/checkpoint_000977'

    irat_0='exp_IRAT/IRAT_Inter_30agents_(no-grad_clip)(no-norm_adv)(mean_nei_r)/IRATTrainer_Intersection_4f15c_00000_0_neighbours_distance=40,num_agents=30,start_seed=5000,nei_rewards_add_coeff=1,nei_rewards_mo_2023-08-10_12-49-37/checkpoint_001172'

    irat_1='exp_IRAT/IRAT_Inter_30agents_BEST(0,1)(10,1)/IRATTrainer_Intersection_14445_00004_4_neighbours_distance=40,num_agents=30,start_seed=5000,nei_rewards_add_coeff=0.1000,nei_rewar_2023-08-11_22-40-55/checkpoint_001465'

    irat_best_aiboy='exp_IRAT/IRAT_Inter_30agents_BEST(0,1)(10,1)/IRATTrainer_Intersection_be499_00000_0_neighbours_distance=40,num_agents=30,start_seed=5000,nei_rewards_add_coeff=1,nei_rewards_mo_2023-08-10_13-14-12/checkpoint_000782'


    irat_best_aiboy_2='exp_IRAT/IRAT_Inter_30agents_BEST(0,1)(10,1)/IRATTrainer_Intersection_e80ff_00000_0_neighbours_distance=40,num_agents=30,start_seed=5000,fcnet_activation=tanh,nei_rewards_add__2023-08-12_14-53-13/checkpoint_001060'

    irat_best_aiboy_86='exp_IRAT/IRAT_Inter_30agents_BEST(0,3)(8,0)/IRATTrainer_Intersection_938ee_00001_1_neighbours_distance=40,num_agents=30,start_seed=5000,fcnet_activation=tanh,nei_rewards_add__2023-08-12_20-34-27/checkpoint_001465'

    ippo_4wokers='exp_IPPO/IPPO_Inter_4agents_(4_workers)(no_norm_adv)/IPPOTrainer_Intersection_3bf41_00000_0_num_agents=30,start_seed=5000,num_others=0_2023-08-11_20-04-33/checkpoint_001172'

    irat_temp='exp_IRAT/IRAT_Inter_30agents_BEST(0,1)(10,0)_4/IRATTrainer_Intersection_e4cf1_00006_6_entropy_coeff=0,neighbours_distance=40,num_agents=30,start_seed=5000,1=0.5000,nei_rewards_a_2023-08-15_13-16-30/checkpoint_000900'
    

    # checkpoint_path = ScCO_30a_5000_intersection_no_others_nei_dis_10m
    checkpoint_path = ScCO_30a_5000_intersection_no_others_nei_dis_10m_max_r_init_0
    # checkpoint_path = ScCO_30a_5000_intersection_no_others_nei_dis_10m_coeff_1
    # checkpoint_path = ippo_16a_4others
    # checkpoint_path = ippo_30a_obs_91_2
    # checkpoint_path = pth
    # checkpoint_path = ippo_best
    # checkpoint_path = SoCO_DC
    # checkpoint_path = max_ego_and_nei_norm_a
    # checkpoint_path = ppp
    checkpoint_path = irat_np_clip
    checkpoint_path = irat_0
    # checkpoint_path = irat_1
    checkpoint_path = irat_best_aiboy_2
    checkpoint_path = irat_best_aiboy_86
    # checkpoint_path = irat_temp
    # checkpoint_path = ippo_4wokers
    algo = Algorithm.from_checkpoint(checkpoint_path)

    return algo


def compute_actions_for_multi_agents_separately(algo, obs, state=None):
    actions = {}
    for agent_id in obs:
        o = obs[agent_id]
        if state:
            o = [o]
            action, state_out, _ = algo.compute_single_action(o, state=state, explore=False)
            action, state_out, _ = algo.compute_actions(o, state=state, explore=False)
            print(action)
            print(state_out)
        else:
            actions[agent_id] = algo.compute_single_action(o, state=state, explore=False)
    return actions


def compute_actions_for_multi_agents_in_batch(algo, obs, infos, state=None):
    info_list = []
    assert isinstance(infos, dict)
    for k in obs:
        infos[k]['agent_id'] = k
        info_list.append(infos[k])
        
    state = [np.reshape(state[0], (1, -1))]
   
    # obs = flatten_obs_dict(obs)
    actions = algo.compute_actions(obs, state=state, info=info_list, explore=False)
    print(actions)
    return actions


def compute_actions(algo: Algorithm, obs, extra_actions=False):
    if extra_actions:
        actions, states, infos = algo.compute_actions(observations=obs, full_fetch=True)
        return actions, infos
    else:
        actions = algo.compute_actions(observations=obs)
        return actions



if __name__ == "__main__":
    from metadrive.component.vehicle.base_vehicle import BaseVehicle
    from env.env_wrappers import get_rllib_compatible_env, get_neighbour_env
    from env.env_utils import get_metadrive_ma_env_cls

    env_name, env_cls = get_rllib_compatible_env(
                        get_neighbour_env(
                        get_metadrive_ma_env_cls(SCENE)), 
                        return_class=True)

    # env_name, env_cls = get_rllib_compatible_env(
    #                         get_tracking_md_env(
    #                             get_metadrive_ma_env_cls(SCENE)
    #                         ), return_class=True)

   # === Environmental Setting ===
    env_config = dict(
        use_render=False,
        num_agents=30,    
        allow_respawn=True,
        return_single_space=True,
        # crash_done=True,
        # delay_done=0,
        start_seed=5000,
        delay_done=25,
        vehicle_config=dict(
            lidar=dict(
                num_lasers=72, 
                distance=40, 
                num_others=0,
            )
        ),
        # == neighbour config ==
        # use_dict_obs=False,
        # add_compact_state=False, # add BOTH ego- & nei- compact-state simultaneously
        # add_nei_state=False,
        # num_neighbours=4,
        # neighbours_distance=40,
    )
    
    algo = get_algo_new()

    # === init metric callbacks ===
    callbacks = MetricCallbacks()

    env = env_cls(env_config)
    obs, infos = env.reset()
    callbacks.on_episode_start()

    stop_render = False
    NUM_EPISODES_TOTAL = 10
    cur_epi = 0

    RENDER = False
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
        "num_success": 0,
        "total_agents": 0
    }

    epi_succ_rate_list = []
    epi_crash_rate_list = []
    epi_out_rate_list = []
    epi_max_step_rate_list = []
    total_agents_list = []

    epi_neighbours_list = defaultdict(list)

    while not stop_render:
        
        # if ALGO == 'ARPPO' or ALGO == 'ARCCPPO_concat':
        # actions = compute_actions_for_multi_agents_in_batch(algo, obs, infos, state=state)
        # else:
        # actions = compute_actions_for_multi_agents_separately(algo, obs, state=state)
        # actions = compute_actions(algo, obs, extra_actions=False)
        HAS_SVO = False
        if HAS_SVO:
            actions, infos = compute_actions(algo, obs, extra_actions=True)
            svo = infos['svo'] # ndarray (N_agents, )
        else:
            actions = compute_actions(algo, obs, extra_actions=False)
            svo = None

        obs, rew, term, trunc, infos = env.step(actions)
        callbacks.on_episode_step(infos=infos)

        # on_episode_step()
        for a, info in infos.items():
            epi_neighbours_list[a].append(len(info['neighbours']))


        # === episode end ===
        if term['__all__']:
            cur_epi += 1
            callbacks.on_episode_end(metrics)
            epi_succ_rate_list.append(metrics["success_rate"])
            epi_crash_rate_list.append(metrics["crash_rate"])
            epi_out_rate_list.append(metrics["out_of_road_rate"])
            epi_max_step_rate_list.append(metrics["max_step_rate"])
            total_agents_list.append(metrics["total_agents"])

            env_steps = metrics["env_steps"]
            num_success = metrics["num_success"]
            num_failures = metrics["total_agents"] - num_success
            efficiency = (num_success - num_failures) / env_steps * 100
            metrics['efficiency'] = efficiency    

            printPanel(metrics, f'episode {cur_epi}/{NUM_EPISODES_TOTAL} ends.')

            all_mean_nei = []
            max_nei = 0
            for a, nei_l in epi_neighbours_list.items():
                max_nei = max(max_nei, np.max(nei_l))
                all_mean_nei.append(np.mean(nei_l))
            print('mean num_neighbours', np.mean(all_mean_nei))
            print('max num_neighbours', np.max(all_mean_nei))
            print('min num_neighbours', np.min(all_mean_nei))
            print('max_nei', max_nei)
            

            if cur_epi >= NUM_EPISODES_TOTAL:
                stop_render = True
                break

            callbacks.on_episode_start()
            obs, infos = env.reset()
           
        if RENDER:
            # vehicles = env.vehicles_including_just_terminated
            vehicles = env.vehicles
            for agent in actions:
                break
            if agent in vehicles:
                v: BaseVehicle = vehicles[agent]
                color = v.panda_color
            else:
                v = None
                color = None

            text = {
                'id ': f'{agent}',
                'color': color,
                # position=vehicle.position,
            }

            if svo is not None:
                svo_args = {
                    'svo': svo[0],
                    'cos': np.cos(svo[0]),
                    'sin': np.sin(svo[0]),
                }
                text.update(svo_args)

            env.render(
                text=text,
                current_track_vehicle = None,
                mode="top_down", 
                film_size=(1000, 1000),
                screen_size=(1000, 1000),
            )

    env.close()
    print_final_summary((
        epi_succ_rate_list, 
        epi_crash_rate_list, 
        epi_out_rate_list, 
        epi_max_step_rate_list
    ))

    algo.stop()