from collections import defaultdict
import pickle
from typing import Dict, Tuple

import numpy as np

from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_trainable, register_env
    
from metadrive import (
    MultiAgentRoundaboutEnv, MultiAgentIntersectionEnv, MultiAgentTollgateEnv,
    MultiAgentBottleneckEnv, MultiAgentParkingLotEnv
)
from metadrive.component.vehicle.base_vehicle import BaseVehicle

from env.env_wrappers import get_rllib_compatible_env, get_tracking_md_env
from env.env_copo import get_lcf_env, get_lcf_from_checkpoint
from env.env_wrappers import get_rllib_compatible_env, get_neighbour_env
from env.env_utils import get_metadrive_ma_env_cls
from utils.debug import print, printPanel
from draw_trajectory import get_single_frame


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
        metrics["success_rate"] = np.mean(arrive_dest_list).round(4)
        metrics["total_agents"] = len(arrive_dest_list)
        metrics["num_success"] = np.count_nonzero(arrive_dest_list)
        metrics["num_crash"] = np.count_nonzero(crash_list)
        metrics["crash_rate"] = np.mean(crash_list).round(4)
        metrics["out_of_road_rate"] = np.mean(out_of_road_list).round(4)
        metrics["max_step_rate"] = np.mean(max_step_list).round(4)

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
        rates.append(np.mean(rate_list).round(4))

    # res = []
    prefix_strs = ('succ_rate', 'crash_rate', 'out_rate', 'maxstep_rate')
    for prefix, rate in zip(prefix_strs, rates):
        res = prefix+': '+str(rate*100)+'%'
        print(res, end='\n')
    

def compute_actions(algo: Algorithm, obs, extra_actions=False):
    if extra_actions:
        actions, states, infos = algo.compute_actions(observations=obs, full_fetch=True)
        return actions, infos
    else:
        actions = algo.compute_actions(observations=obs)
        return actions


def get_ckp() -> Tuple:
    copo_inter2='exp_CoPO/CoPO_Inter_30agents_1M/CoPOTrainer_Intersection_16951_00000_0_start_seed=7000_2023-09-26_00-39-25/checkpoint_000977'

    scpo_inter_88 = 'exp_SCPO/BEST_Backup/AIBOY/SCPOTrainer_Intersection_99ed5_00007_7_num_agents=30,start_seed=5000,0=1000000.0000,1=1,1=5,0=800000.0000,1=0.5000_2023-09-23_01-08-07/checkpoint_001450' # 88%
    
    scpo_inter_89 = 'exp_SCPO/SUCCESS/SCPO_Inter_30agents_2M_(8workers)_2e6_8seeds/SCPOTrainer_Intersection_f9646_00001_1_num_agents=30,start_seed=5000,1=2,team_clip_param=0.3000,1=3,1=1_2023-10-15_01-13-20/checkpoint_001954'

    ippo_inter = 'eval_checkpoints/ippo/inter/IPPOTrainer_Intersection_1de81_00002_2_start_seed=7000,lr=0.0000_2023-10-14_20-06-33/checkpoint_001941'

    ippo_round = 'eval_checkpoints/ippo/round/IPPOTrainer_Roundabout_134ec_00000_0_start_seed=5000,lr=0.0000_2023-11-02_15-47-30/checkpoint_000770'

    ccppo_concat_inter = 'eval_checkpoints/ccppo-concat/inter/CCPPOTrainer_Intersection_2a734_00001_1_start_seed=6000,fuse_mode=concat_2023-10-19_15-43-16/checkpoint_001950'

    ccppo_concat_round = 'eval_checkpoints/ccppo-concat/round/CCPPOTrainer_Roundabout_103c8_00007_7_start_seed=12000,fuse_mode=concat_2023-11-17_02-43-41/checkpoint_000925'


    ccppo_mf_inter = 'eval_checkpoints/ccppo-mf/inter/CCPPOTrainer_Intersection_86cef_00000_0_start_seed=7000,fuse_mode=mf_2023-10-19_21-43-46/checkpoint_001900'

    ccppo_mf_round = 'eval_checkpoints/ccppo-mf/round/CCPPOTrainer_Roundabout_86716_00000_0_start_seed=5000,fuse_mode=mf_2023-11-04_20-49-00/checkpoint_001900'

    ccppo_inter = 'exp_CCPPO/CCPPO_Inter_30agents_concat/CCPPOTrainer_Intersection_fd7fd_00000_0_start_seed=5000_2023-09-23_14-18-19/checkpoint_001320'

    ccppo_round = 'exp_CCPPO/CCPPO_Round_40agents_8workers_2M_concat+mf_8seeds_lr=3e-5/CCPPOTrainer_Roundabout_f1bb2_00008_8_start_seed=5000,fuse_mode=concat_2023-10-20_22-42-50/checkpoint_001880'

    copo_inter = 'eval_checkpoints/copo/inter/CoPOTrainer_Intersection_5f02d_00000_0_start_seed=12000,lr=0.0000_2023-11-03_22-22-08/checkpoint_001220'

    copo_round = 'eval_checkpoints/copo/round/CoPOTrainer_Roundabout_f9e4d_00000_0_start_seed=5000_2023-11-05_19-17-59/checkpoint_001890'

    scpo_round = 'eval_checkpoints/scpo/round/SCPOTrainer_Roundabout_c407d_00000_0_start_seed=8000,0=2000000.0000,1=1,nei_rewards_add_coeff=1,team_clip_param=0.3000,1=4,0=20000_2023-11-24_17-13-43/checkpoint_000858'

    ippo_rs_inter = 'exp_IPPO-RS/IPPO-RS_Inter_30agents_8workers_phi=1/IPPORSTrainer_Intersection_cbe55_00002_2_start_seed=7000,lr=0.0000_2024-01-04_22-23-14/checkpoint_001954' # 91%
    ippo_rs_inter_2 = 'exp_IPPO-RS/IPPO-RS_Inter_30agents_8workers_phi=1/IPPORSTrainer_Intersection_cbe55_00000_0_start_seed=5000,lr=0.0000_2024-01-04_22-23-14/checkpoint_001954' # 40%

    # checkpoint_path = scpo_inter_88
    # checkpoint_path = scpo_inter_89
    # checkpoint_path = ippo_inter
    # checkpoint_path = ippo_round
    # checkpoint_path = ccppo_concat_inter
    # checkpoint_path = ccppo_concat_round
    # checkpoint_path = ccppo_mf_inter
    # checkpoint_path = ccppo_mf_round
    # checkpoint_path = ccppo_inter
    # checkpoint_path = ccppo_round
    # checkpoint_path = copo_inter
    # checkpoint_path = copo_round
    # checkpoint_path = scpo_round
    checkpoint_path = ippo_rs_inter_2

    if 'ippo' in checkpoint_path.lower():
        algo_name = 'ippo'
    elif 'ccppo-concat' in checkpoint_path.lower():
        algo_name = 'mappo'
    elif 'ccppo-mf' in checkpoint_path.lower():
        algo_name = 'mfpo'
    elif 'scpo' in checkpoint_path.lower():
        algo_name = 'scpo'
    elif 'copo' in checkpoint_path.lower():
        algo_name = 'copo'
    else:
        raise NotImplementedError

    if 'inter' in checkpoint_path.lower():
        scene_name = 'inter'
    elif 'round' in checkpoint_path.lower():
        scene_name = 'round'
    else:
        raise NotImplementedError

    return checkpoint_path, algo_name, scene_name, 'copo' in checkpoint_path.lower()


def get_algo(ckp):
    algo = Algorithm.from_checkpoint(ckp)
    return algo


def get_screen_sizes():
    try: 
        from screeninfo import get_monitors
        monitors = get_monitors() 
        if len(monitors) == 1 and monitors[0].height == 982:
            return (800, 800)
        else:
            return (1000, 1000)
    except:
        print('Try `pip install screeninfo`!')
        return (1000, 1000)


if __name__ == "__main__":
    
    # === Set a lot of configs ===

    # SCENE = "intersection"
    # SCENE = "roundabout"
    NUM_EPISODES_TOTAL = 20
    # RENDER = False
    RENDER = True
    # RECORD_TRAJ = True
    RECORD_TRAJ = False
    # RECORD_RENDER_IMAGE = True
    RECORD_RENDER_IMAGE = False
    if RECORD_RENDER_IMAGE:
        import pygame


    ckp, algo_name, scene, should_wrap_copo = get_ckp() # only copo should be wrapped
    env_cls, env_abbr_name = get_metadrive_ma_env_cls(scene, return_abbr=True) 

    if should_wrap_copo:
        env_cls = get_lcf_env(env_cls)
    else:
        env_cls = get_neighbour_env(env_cls)
    env_name, env_cls = get_rllib_compatible_env(env_cls, return_class=True)

   # === Environmental Setting ===
    env_config = dict(
        use_render=False,
        # num_agents=40,    
        horizon=1000,
        allow_respawn=True,
        return_single_space=True,
        # crash_done=True,
        # delay_done=0,
        # start_seed=5000,
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
        # add_nei_state=False,
        # num_neighbours=4,
        # neighbours_distance=40,
    )
    
    algo = get_algo(ckp)
    print('=== Testing ', algo_name, env_abbr_name)

    # === init metric callbacks ===
    callbacks = MetricCallbacks()

    env = env_cls(env_config)
    if should_wrap_copo:
        lcf_mean, lcf_std = get_lcf_from_checkpoint(ckp)
        print('=== get lcf mean/std:', lcf_mean, lcf_std)
        env.set_lcf_dist(lcf_mean, lcf_std)

    obs, infos = env.reset()
    print('=== start_seed:', env.config['start_seed'])

    callbacks.on_episode_start()

    stop_render = False
    cur_step = 0
    cur_epi = 0

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

    frame = []


    while not stop_render:
        
        actions = compute_actions(algo, obs, extra_actions=False)

        obs, r, tm, tc, infos = env.step(actions)
        callbacks.on_episode_step(infos=infos)
        cur_step += 1

        frame.append(get_single_frame(env, tm, infos))

        # on_episode_step()
        for a, info in infos.items():
            epi_neighbours_list[a].append(len(info['neighbours']))


        # === episode end ===
        if tm['__all__']:
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
            
            if RECORD_TRAJ:
                succ = f'{np.mean(epi_succ_rate_list).round(4)*100}%'
                traj = {'frame': frame}
                file_name = f'trajectories/traj/traj_{algo_name}_{env_abbr_name.lower()}_{succ}.pkl'
                with open(file_name, 'wb') as file:
                    pickle.dump(traj, file)
                print('\n=== traj saved at', file_name, '!\n')

            if cur_epi >= NUM_EPISODES_TOTAL:
                stop_render = True
                break

            callbacks.on_episode_start()
            obs, infos = env.reset()
           
           
        if RENDER:
            vehicles = env.vehicles_including_just_terminated
            # vehicles = env.vehicles
            for agent in actions:
                break
            if agent in vehicles and vehicles[agent]:
                v: BaseVehicle = vehicles[agent]
                color = v.panda_color
                position = vehicles[agent].position
            else:
                v = None
                color = None
                position = None

            text = {
                'id ': f'{agent}',
                'color': color,
                'position': f'{position}',
            }
            # print(position)
            # input()

            
            if not RECORD_RENDER_IMAGE:
                ret = env.render(
                    text=None,
                    num_stack=15, # default 15
                    # current_track_vehicle = v,
                    mode="top_down", 
                    # draw_target_vehicle_trajectory=True,
                    film_size=get_screen_sizes(),
                    screen_size=get_screen_sizes(),
                )

            else:
                if scene == 'inter':
                    screen_size = (3000, 3000)
                    film_size = screen_size
                    crop_coeff = 0.45 # remain how many size of width/height
                elif scene == 'round':
                    screen_size = (3000, 3000)
                    film_size = screen_size
                    crop_coeff = 0.5 # remain how many size of width/height
                crop_rect = (
                    screen_size[0]*(1-crop_coeff)/2,
                    screen_size[1]*(1-crop_coeff)/2,
                    screen_size[0]*crop_coeff,
                    screen_size[1]*crop_coeff
                )
                
                ret = env.render(
                    text=None,
                    num_stack=50, # default 15
                    # current_track_vehicle = v,
                    mode="top_down", 
                    # draw_target_vehicle_trajectory=True,
                    film_size=film_size,
                    screen_size=screen_size,
                    draw_future_traj=True,
                    road_color=(35, 35, 35),
                )

                if cur_step > 100 and cur_step < 1000:
                    pygame.image.save(ret.convert_alpha().subsurface(pygame.Rect(crop_rect)), f'trajectories/render/{scene}/{algo_name}_step{cur_step}.png')
                # if cur_step == 1400: exit()


    # env.close()
    print_final_summary((
        epi_succ_rate_list, 
        epi_crash_rate_list, 
        epi_out_rate_list, 
        epi_max_step_rate_list
    ))

    algo.stop()