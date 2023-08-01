from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
from collections import defaultdict
import numpy as np

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.evaluation import RolloutWorker


class MultiAgentDrivingCallbacks(DefaultCallbacks):

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs) -> None:
        # actions = samples['default_policy']['actions']
        # print('[on_sample_end] actions:', len(actions), actions, sep='\n')
        pass


    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        # == need aggragating ==
        episode.user_data["velocity"] = defaultdict(list)
        episode.user_data["steering"] = defaultdict(list)
        episode.user_data["step_reward"] = defaultdict(list) # {'agent0': [0.1, 1.2, ...], ...}
        episode.user_data["acceleration"] = defaultdict(list)
        # episode.user_data["cost"] = defaultdict(list)
        episode.user_data["neighbours"] = defaultdict(list)
        # == accumulative, only need last one ==
        # episode.user_data["agent_steps"] = defaultdict(list) # of single agent
        # episode.user_data["agent_rewards"] = defaultdict(list) 

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        '''
        base_env: <class 'ray.rllib.env.multi_agent_env.MultiAgentEnvWrapper'>
        base_env.envs[env_index]: <RLlibMultiAgentMetaDrive<MultiAgentRoundaboutEnv instance>>
        '''

        active_keys = list(base_env.envs[env_index].vehicles.keys())

        # The agent_rewards dict contains all agents' reward, not only the active agent!
        # active_keys = [k for k, _ in episode.agent_rewards.keys()]

        for agent_id in active_keys:
            k = agent_id
            info = episode._last_infos.get(k)
            if info:
                if "step_reward" not in info:
                    continue
                episode.user_data["velocity"][k].append(info["velocity"])
                episode.user_data["steering"][k].append(info["steering"])
                episode.user_data["step_reward"][k].append(info["step_reward"])
                episode.user_data["acceleration"][k].append(info["acceleration"])
                # episode.user_data["cost"][k].append(info["cost"])
                # episode.user_data["agent_steps"][k].append(info["episode_length"])
                # episode.user_data["agent_rewards"][k].append(info["episode_reward"])
                episode.user_data["neighbours"][k].append(len(info.get("neighbours", [])))

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        # keys = [k for k, _ in episode.agent_rewards.keys()]
        agents = episode.get_agents()

        arrive_dest_list = []
        crash_list = []
        out_of_road_list = []
        max_step_list = []

        # Newly introduced metrics
        # current_distance_list = []

        for k in agents:
            info = episode._last_infos[k]

            # == Newly introduced metrics ==
            # current_distance = info.get("current_distance", -1)
            # current_distance_list.append(current_distance)

            # == Rate ==
            arrive_dest = info.get("arrive_dest", False)
            crash = info.get("crash", False)
            out_of_road = info.get("out_of_road", False)
            max_step = not (arrive_dest or crash or out_of_road)
            # assert max_step == info.get("max_step") # not approved all the time!
            arrive_dest_list.append(arrive_dest)
            crash_list.append(crash)
            out_of_road_list.append(out_of_road)
            max_step_list.append(max_step)


        # episode.custom_metrics["current_distance"] = np.mean(current_distance_list)
        episode.custom_metrics["success_rate"] = np.mean(arrive_dest_list)
        episode.custom_metrics["crash_rate"] = np.mean(crash_list)
        episode.custom_metrics["out_of_road_rate"] = np.mean(out_of_road_list)
        episode.custom_metrics["max_step_rate"] = np.mean(max_step_list)

        # 在一个episode结束后， 计算user_data项目中平均每个agent数据的最大、最小、平均值
        for info_k, info_dict in episode.user_data.items(): # info_dict: {'agent0': [...], ...}
            self._add_item(episode, info_k, [vv for v in info_dict.values() for vv in v]) 
        
        # 对于 accumulative 的数值，只计算每个agent最后一步的值即可
        for info_k in ['episode_length', 'episode_reward', 'episode_distance']:
            self._add_item(episode, info_k, [episode._last_infos[k][info_k] for k in agents])

        episode.custom_metrics["environmental_agents_rewards"] = np.sum(
            [episode._last_infos[a]['episode_reward'] for a in agents]
        ) # 一个episode中所有agent的episode_reward的总和)
        episode.custom_metrics["num_agents_total"] = len(episode.get_agents())
        # assert len(episode.get_agents()) == len(episode._last_infos.keys()), (episode.get_agents(), episode._last_infos.keys())

    def _add_item(self, episode, name, value_list):
        episode.custom_metrics["{}_max".format(name)] = float(np.max(value_list))
        episode.custom_metrics["{}_mean".format(name)] = float(np.mean(value_list))
        episode.custom_metrics["{}_min".format(name)] = float(np.min(value_list))


    def on_postprocess_trajectory(
        self,
        *,
        worker: "RolloutWorker",
        episode: Episode,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: Dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None: 
        pass
        

    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
        # 记录 svo 值
        if 'svo' in train_batch:
            svo = train_batch['svo']
            result['svo_mean'] = np.mean(svo)
            result['svo_max'] = np.max(svo)
            result['svo_min'] = np.min(svo)
            result['svo_std'] = np.std(svo)


    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: dict,
        **kwargs,
    ) -> None:
        num_envs = 1 * algorithm.config.num_rollout_workers
        result["SuccessRate"] = np.nan
        result["CrashRate"] = np.nan
        result["OutRate"] = np.nan
        result["MaxStepRate"] = np.nan
        result["RewardMean"] = np.nan
        result["StepMean"] = np.nan
        result["DistanceMean"] = np.nan
        result["NeighboursMean"] = np.nan
        result["RewardsTotal"] = np.nan
        result["AgentsTotal"] = np.nan
        if "success_rate_mean" in result["custom_metrics"]:
            result["SuccessRate"] = result["custom_metrics"]["success_rate_mean"]
            result["CrashRate"] = result["custom_metrics"]["crash_rate_mean"]
            result["OutRate"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["MaxStepRate"] = result["custom_metrics"]["max_step_rate_mean"]

            result["RewardMean"] = result["custom_metrics"]["episode_reward_mean_mean"]
            result["StepMean"] = result["custom_metrics"]["episode_length_mean_mean"]
            result["DistanceMean"] = result["custom_metrics"]["episode_distance_mean_mean"]
            result["NeighboursMean"] = result["custom_metrics"]["neighbours_mean_mean"] 

            result["RewardsTotal"] = result["custom_metrics"]["environmental_agents_rewards_mean"] 
            result["AgentsTotal"] = result["custom_metrics"]["num_agents_total_mean"] 

        # from utils.debug import printPanel
        # printPanel(result)

