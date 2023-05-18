from gymnasium.spaces import Space
import logging
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.evaluation.collectors.agent_collector import AgentCollector
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, concat_samples
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.spaces.space_utils import get_dummy_batch_for_space
from ray.rllib.utils.typing import (
    AgentID,
    EpisodeID,
    EnvID,
    PolicyID,
    TensorType,
    ViewRequirementsDict,
)
from ray.util.debug import log_once

_, tf, _ = try_import_tf()
torch, _ = try_import_torch()


logger = logging.getLogger(__name__)

from rich import print, inspect


class MultiAgentARSampleCollector(SimpleListCollector):

    def __init__(
        self, 
        policy_map: PolicyMap, 
        clip_rewards: Union[bool, float], 
        callbacks: DefaultCallbacks, 
        multiple_episodes_in_batch: bool = True, 
        rollout_fragment_length: int = 200, 
        count_steps_by: str = "env_steps"
    ):

        super().__init__(
            policy_map,
            clip_rewards,
            callbacks,
            multiple_episodes_in_batch,
            rollout_fragment_length,
            count_steps_by,
        )

        print('>>> [MultiAgentARSampleCollector] begin init()...')


    

    @override(SimpleListCollector)
    def get_inference_input_dict(self, policy_id: PolicyID) -> Dict[str, TensorType]:
        exit()
        print('>>> [MultiAgentARSampleCollector] begin get_inference_input_dict()...')

        policy = self.policy_map[policy_id]
        keys = self.forward_pass_agent_keys[policy_id]
        batch_size = len(keys)

        # Return empty batch, if no forward pass to do.
        if batch_size == 0:
            return SampleBatch()

        buffers = {}
        for k in keys:
            collector = self.agent_collectors[k]
            buffers[k] = collector.buffers
        # Use one agent's buffer_structs (they should all be the same).
        buffer_structs = self.agent_collectors[keys[0]].buffer_structs

        input_dict = {}
        for view_col, view_req in policy.view_requirements.items():
            # Not used for action computations.
            if not view_req.used_for_compute_actions:
                continue

            # Create the batch of data from the different buffers.
            data_col = view_req.data_col or view_col
            delta = (
                -1
                if data_col
                in [
                    SampleBatch.OBS,
                    SampleBatch.INFOS,
                    SampleBatch.ENV_ID,
                    SampleBatch.EPS_ID,
                    SampleBatch.AGENT_INDEX,
                    SampleBatch.T,
                ]
                else 0
            )
            # Range of shifts, e.g. "-100:0". Note: This includes index 0!
            if view_req.shift_from is not None:
                time_indices = (view_req.shift_from + delta, view_req.shift_to + delta)
            # Single shift (e.g. -1) or list of shifts, e.g. [-4, -1, 0].
            else:
                time_indices = view_req.shift + delta

            # Loop through agents and add up their data (batch).
            data = None
            for k in keys:
                # Buffer for the data does not exist yet: Create dummy
                # (zero) data.
                if data_col not in buffers[k]:
                    if view_req.data_col is not None:
                        space = policy.view_requirements[view_req.data_col].space
                    else:
                        space = view_req.space

                    if isinstance(space, Space):
                        fill_value = get_dummy_batch_for_space(
                            space,
                            batch_size=0,
                        )
                    else:
                        fill_value = space

                    self.agent_collectors[k]._build_buffers({data_col: fill_value})

                if data is None:
                    data = [[] for _ in range(len(buffers[keys[0]][data_col]))]

                # `shift_from` and `shift_to` are defined: User wants a
                # view with some time-range.
                if isinstance(time_indices, tuple):
                    # `shift_to` == -1: Until the end (including(!) the
                    # last item).
                    if time_indices[1] == -1:
                        for d, b in zip(data, buffers[k][data_col]):
                            d.append(b[time_indices[0] :])
                    # `shift_to` != -1: "Normal" range.
                    else:
                        for d, b in zip(data, buffers[k][data_col]):
                            d.append(b[time_indices[0] : time_indices[1] + 1])
                # Single index.
                else:
                    for d, b in zip(data, buffers[k][data_col]):
                        d.append(b[time_indices])

            np_data = [np.array(d) for d in data]
            if data_col in buffer_structs:
                input_dict[view_col] = tree.unflatten_as(
                    buffer_structs[data_col], np_data
                )
            else:
                input_dict[view_col] = np_data[0]

        self._reset_inference_calls(policy_id)

        return SampleBatch(
            input_dict,
            seq_lens=np.ones(batch_size, dtype=np.int32)
            if "state_in_0" in input_dict
            else None,
        )
