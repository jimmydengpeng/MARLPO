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
        # logger.debug('base_env', type(base_env)) 
        episode.user_data["velocity"] = defaultdict(list)
        episode.user_data["steering"] = defaultdict(list)
        episode.user_data["step_reward"] = defaultdict(list) # {'agent0': [0, 1, 1.2, 0.2, ...], ...}
        episode.user_data["acceleration"] = defaultdict(list)
        episode.user_data["cost"] = defaultdict(list)
        episode.user_data["episode_length"] = defaultdict(list)
        episode.user_data["episode_reward"] = defaultdict(list)
        episode.user_data["num_neighbours"] = defaultdict(list)


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
            # info = episode.last_info_for(k) # deprecated!
            info = episode._last_infos.get(k)
            if info:
                if "step_reward" not in info:
                    continue
                episode.user_data["velocity"][k].append(info["velocity"])
                episode.user_data["steering"][k].append(info["steering"])
                episode.user_data["step_reward"][k].append(info["step_reward"])
                episode.user_data["acceleration"][k].append(info["acceleration"])
                episode.user_data["cost"][k].append(info["cost"])
                episode.user_data["episode_length"][k].append(info["episode_length"])
                episode.user_data["episode_reward"][k].append(info["episode_reward"])
                episode.user_data["num_neighbours"][k].append(len(info.get("neighbours", [])))

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
        keys = [k for k, _ in episode.agent_rewards.keys()]
        arrive_dest_list = []
        crash_list = []
        out_of_road_list = []
        max_step_list = []

        # # Newly introduced metrics
        # track_length_list = []
        # route_completion_list = []
        current_distance_list = []

        for k in keys:
            # info = episode.last_info_for(k)
            info = episode._last_infos[k]

            # == Newly introduced metrics ==
            # route_completion = info.get("route_completion", -1)
            # track_length = info.get("track_length", -1)
            current_distance = info.get("current_distance", -1)
            current_distance_list.append(current_distance)
            # track_length_list.append(track_length)
            # route_completion_list.append(route_completion)

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

        # Newly introduced metrics
        # episode.custom_metrics["track_length"] = np.mean(track_length_list)
        episode.custom_metrics["current_distance"] = np.mean(current_distance_list)
        # episode.custom_metrics["route_completion"] = np.mean(route_completion_list)

        episode.custom_metrics["success_rate"] = np.mean(arrive_dest_list)
        episode.custom_metrics["crash_rate"] = np.mean(crash_list)
        episode.custom_metrics["out_of_road_rate"] = np.mean(out_of_road_list)
        episode.custom_metrics["max_step_rate"] = np.mean(max_step_list)

        for info_k, info_dict in episode.user_data.items():
            # 记录所有user_data项目中所有agent数据的最大、最小、平均值
            if info_k in ['episode_length', 'episode_reward']: continue
            self._add_item(episode, info_k, [vv for v in info_dict.values() for vv in v]) # info_dict:  {'agent0': [0, 1, 1.2, 0.2, ...], ...}

        episode.custom_metrics["episode_length"] = np.mean(
            [ep_len[-1] for ep_len in episode.user_data["episode_length"].values()]
        ) # 所有agent的episode_length的平均
        episode.custom_metrics["episode_reward"] = np.mean(
            [ep_r[-1] for ep_r in episode.user_data["episode_reward"].values()]
        ) # 所有agent的episode_reward的平均)
        episode.custom_metrics["environment_reward_total"] = np.sum(
            [ep_r[-1] for ep_r in episode.user_data["episode_reward"].values()]
        ) # 一个episode中所有agent的episode_reward的总和)

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
        # pass
        # infos = train_batch[SampleBatch.INFOS]
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
        result["SuccessRate"] = np.nan
        result["CrashRate"] = np.nan
        result["OutRate"] = np.nan
        result["MaxStepRate"] = np.nan
        result["LengthMean"] = result["episode_len_mean"]
        result["DistanceMean"] = np.nan
        if "success_rate_mean" in result["custom_metrics"]:
            result["SuccessRate"] = result["custom_metrics"]["success_rate_mean"]
            result["CrashRate"] = result["custom_metrics"]["crash_rate_mean"]
            result["OutRate"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["MaxStepRate"] = result["custom_metrics"]["max_step_rate_mean"]
            result["RewardMean"] = result["custom_metrics"]["episode_reward_mean"]
            result["DistanceMean"] = result["custom_metrics"]["current_distance_mean"] #TODO
        result["CostMean"] = np.nan
        if "episode_cost_mean" in result["custom_metrics"]:
            result["CostMean"] = result["custom_metrics"]["episode_cost_mean"]

        # present the agent-averaged reward.
        # result["RewardMean"] = result["episode_reward_mean"]
        # result["_episode_policy_reward_mean"] = np.mean(list(result["policy_reward_mean"].values()))
        # result["environment_reward_total"] = np.sum(list(result["policy_reward_mean"].values()))

