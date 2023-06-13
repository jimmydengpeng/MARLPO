from collections import defaultdict
import functools, random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type, Union

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.connectors.connector import Connector, ConnectorContext
from ray.rllib.connectors.util import create_connectors_for_policy
from ray.rllib.core.rl_module import RLModule
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import (
    DeveloperAPI,
    OverrideToImplementCustomLogic,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
    is_overridden,
    override,
)
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TrainerConfigDict

from marlpo.algo_ippo import IPPOTrainer, IPPOConfig
from marlpo.connectors import ModEnvInfoAgentConnector
from marlpo.models.ar_model import ARModel
from marlpo.utils.debug import print, printPanel


ModelCatalog.register_custom_model("ar_model", ARModel)

from rich import print, inspect

torch, nn = try_import_torch()

CENTRALIZED_CRITIC_OBS = "centralized_critic_obs"
COUNTERFACTUAL = "counterfactual"
NEIGHBOUR_INFOS = "neighbour_infos"
NEIGHBOUR_ACTIONS = "neighbour_actions"
EXECUTION_MASK = 'execution_mask'

class ARCCPPOConfig(IPPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or ARCCPPOTrainer)
        self.counterfactual = True
        self.num_neighbours = 4
        self.fuse_mode = "mf"  # In ["concat", "mf", "none"]
        self.mf_nei_distance = 10
        self.old_value_loss = True
        self.random_order = True # [True, False]
        self.edge_descending = True # [True, False, None]
        # self.update_from_dict({"model": {"custom_model": "arcc_model"}})

    def validate(self):
        super().validate()
        assert self["fuse_mode"] in ["mf", "concat", "none"]

        self.model["custom_model_config"]["fuse_mode"] = self["fuse_mode"]
        self.model["custom_model_config"]["counterfactual"] = self["counterfactual"]
        self.model["custom_model_config"]["num_neighbours"] = self["num_neighbours"]


def get_centralized_critic_obs_dim(
    observation_space_shape, action_space_shape, counterfactual, num_neighbours, fuse_mode
):
    """Get the centralized critic"""
    if fuse_mode == "concat":
        pass
    elif fuse_mode == "mf":
        num_neighbours = 1
    elif fuse_mode == "none":  # Do not use centralized critic
        num_neighbours = 0
    else:
        raise ValueError("Unknown fuse mode: ", fuse_mode)
    num_neighbours += 1
    centralized_critic_obs_dim = num_neighbours * observation_space_shape.shape[0]
    if counterfactual:  # Do not include ego action!
        centralized_critic_obs_dim += (num_neighbours - 1) * action_space_shape.shape[0]
    return centralized_critic_obs_dim



def concat_ccppo_process(policy, sample_batch, other_agent_batches, odim, adim, other_info_dim):
    """Concat the neighbors' observations"""
    for index in range(sample_batch.count):

        environmental_time_step = sample_batch["t"][index]

        # "neighbours" may not be in sample_batch['infos'][index]:
        # neighbours = sample_batch['infos'][index]["neighbours"]
        neighbours = sample_batch['infos'][index].get("neighbours", [])
        # print(sample_batch.keys())
        # print('neighbours:', neighbours)
        # print(other_agent_batches)
        # print('concat_ccppo_process...')

        # Note that neighbours returned by the environment are already sorted based on their
        # distance to the ego vehicle whose info is being used here.
        for nei_count, nei_name in enumerate(neighbours):
            if nei_count >= policy.config["num_neighbours"]:
                break

            nei_act = None
            nei_obs = None
            if nei_name in other_agent_batches:
                if len(other_agent_batches[nei_name]) == 3:
                    _, _, nei_batch = other_agent_batches[nei_name]
                else:
                    _, nei_batch = other_agent_batches[nei_name]

                match_its_step = np.where(nei_batch["t"] == environmental_time_step)[0]

                if len(match_its_step) == 0:
                    pass
                elif len(match_its_step) > 1:
                    raise ValueError()
                else:
                    new_index = match_its_step[0]
                    nei_obs = nei_batch[SampleBatch.CUR_OBS][new_index]
                    nei_act = nei_batch[SampleBatch.ACTIONS][new_index]

            if nei_obs is not None:
                start = odim + nei_count * other_info_dim
                sample_batch[CENTRALIZED_CRITIC_OBS][index, start:start + odim] = nei_obs
                if policy.config[COUNTERFACTUAL]:
                    sample_batch[CENTRALIZED_CRITIC_OBS][index, start + odim:start + odim + adim] = nei_act
                    assert start + odim + adim == start + other_info_dim
                else:
                    assert start + odim == start + other_info_dim
    return sample_batch


def mean_field_ccppo_process(policy, sample_batch, other_agent_batches, odim, adim, other_info_dim):
    """Average the neighbors' observations and probably actions."""
    # Note: Average other's observation might not be a good idea.
    # Maybe we can do some feature extraction before averaging all observations

    assert odim + other_info_dim == sample_batch[CENTRALIZED_CRITIC_OBS].shape[1]
    for index in range(sample_batch.count):

        environmental_time_step = sample_batch["t"][index]

        # if "neighbours" in sample_batch['infos'][index]:
        neighbours = sample_batch['infos'][index].get("neighbours", [])
        neighbours_distance = sample_batch['infos'][index].get("neighbours_distance", [])
        
        obs_list = []
        act_list = []

        for nei_count, (nei_name, nei_dist) in enumerate(zip(neighbours, neighbours_distance)):
            if nei_dist > policy.config["mf_nei_distance"]:
                continue

            nei_act = None
            nei_obs = None
            if nei_name in other_agent_batches:
                # inspect(other_agent_batches)
                if len(other_agent_batches[nei_name]) == 3:
                    _, _, nei_batch = other_agent_batches[nei_name]
                else:
                    _, nei_batch = other_agent_batches[nei_name]

                match_its_step = np.where(nei_batch["t"] == environmental_time_step)[0]

                if len(match_its_step) == 0:
                    pass
                elif len(match_its_step) > 1:
                    raise ValueError()
                else:
                    new_index = match_its_step[0]
                    nei_obs = nei_batch[SampleBatch.CUR_OBS][new_index]
                    nei_act = nei_batch[SampleBatch.ACTIONS][new_index]

            if nei_obs is not None:
                obs_list.append(nei_obs)
                act_list.append(nei_act)

        if len(obs_list) > 0:
            sample_batch[CENTRALIZED_CRITIC_OBS][index, odim:odim + odim] = np.mean(obs_list, axis=0)
            if policy.config[COUNTERFACTUAL]:
                sample_batch[CENTRALIZED_CRITIC_OBS][index, odim + odim:odim + odim + adim] = np.mean(act_list, axis=0)

    return sample_batch


class ARCCPPOPolicy(PPOTorchPolicy):
    count = 0
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        create_connectors_for_policy(self, AlgorithmConfig.from_dict(self.config))

        # self.agent_connectors.append(MyAgentConnector(ConnectorContext.from_policy(self)))
        self.agent_connectors.insert_before(
            "ViewRequirementAgentConnector", 
            ModEnvInfoAgentConnector(ConnectorContext.from_policy(self))
        )

        # inspect(self.agent_connectors)
        
        self.view_requirements[SampleBatch.INFOS].used_for_compute_actions = True
        # self.view_requirements[NEIGHBOUR_INFOS] = ViewRequirement()

        self.last_computed_neighbour_actions = None # shape[BatchSize, NeighbourActionsDim], e.g. (1, 8) or (32, 8)
        self.last_computed_execution_mask = None

    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        return {
            NEIGHBOUR_ACTIONS: self.last_computed_neighbour_actions,
            EXECUTION_MASK: self.last_computed_execution_mask
        }


    @override(PPOTorchPolicy)
    def action_sampler_fn(
        self, 
        model: ModelV2, 
        *, 
        obs_batch: SampleBatch, 
        state_batches: TensorType, 
        **kwargs
    ) -> Tuple[TensorType, TensorType, TensorType, List[TensorType]]:
        """Custom function for sampling new actions given policy.

        Args:
            model: Underlying model.
            obs_batch: Observation tensor batch (type: SampleBatch(dict))
                NOTE: when doing _initialize_loss_from_dummy_batch(), the dummy_batch is used with TorchPolicyV2._lazy_tensor_dict(), meaning that values in this SampleBatch will be type of numpy.ndarray util __getitem__, then the type will be torch.Tensor:
                {
                    SampleBatch.OBS: torch.Size([BatchSize, obs_dim])
                    SampleBatch.INFOS: torch.Size([BatchSize]) of 0(init) or dict(rollout)
                    NEIGHBOUR_INFOS: torch.Size([BatchSize])
                    NEIGHBOUR_ACTIONS: torch.Size([BatchSize, 8])
                } 
                NOTE: BatchSize:
                            [32]                    when initializing
                            [num_agents_per_step]   when rollout
                            [training_batch_size]   when training

            state_batches: Action sampling state batch.
            **kwargs: {'explore': bool, 'timestep': int}

        Returns:
            Sampled action
            Log-likelihood
            # Action distribution inputs
            Updated state
        """
        input_dict = obs_batch
        assert state_batches == [] or state_batches == None
        assert not is_overridden(self.action_distribution_fn)
        assert SampleBatch.INFOS in input_dict
        infos = input_dict[SampleBatch.INFOS]
        dist_class = self.dist_class
        num_neighbours = self.config.get('num_neighbours')
        # self.last_computed_neighbour_actions = None # 重置


        # == print input msg ==
        _in_msg={}
        _in_msg['input_dict'] = input_dict
        _in_msg['input_dict.type'] = type(input_dict)
        _in_msg['input_dict.is_training'] = input_dict.is_training
        _in_msg[SampleBatch.INFOS+'.type'] = type(infos)
        if isinstance(infos, np.ndarray) or isinstance(infos, torch.Tensor):
            _in_msg[SampleBatch.INFOS+'.shape'] = infos.shape
            if isinstance(infos[0], dict): 
                _in_msg['all_agents'] = []
                for idx, info in enumerate(infos):
                   _in_msg['all_agents'].append(info.get('agent_id', f'dummy_agent{idx}'))
            # msg[SampleBatch.INFOS+'[0]'] = infos[0]

        # msg['NEIGHBOUR_INFOS.type'] = type(input_dict[NEIGHBOUR_INFOS])
        # msg['NEIGHBOUR_INFOS.shape'] = input_dict[NEIGHBOUR_INFOS].shape
        # msg['NEIGHBOUR_INFOS[0]'] = input_dict[NEIGHBOUR_INFOS][0]
        # msg['NEIGHBOUR_ACTIONS.type'] = type(input_dict[NEIGHBOUR_ACTIONS])
        # msg['NEIGHBOUR_ACTIONS.shape'] = input_dict[NEIGHBOUR_ACTIONS].shape
        # msg['NEIGHBOUR_ACTIONS[0]'] = input_dict[NEIGHBOUR_ACTIONS][0]

        _in_msg.update(kwargs)
        _in_msg['random_order'] = self.config.get('random_order', False)
        # printPanel(_in_msg, title=f'{self.__class__.__name__}.action_sampler_fn()')
        # print('==='*20)
        # for k in input_dict:
        #     print(k, type(input_dict[k]), input_dict[k].shape)
        #     print('---'*20)
        # print('==='*20)

        assert isinstance(infos, np.ndarray) or isinstance(infos, torch.Tensor)

        # === Call the exploration before_compute_actions hook. ===
        explore = kwargs.get('explore', None)
        timestep = kwargs.get('timestep', None)
        self.exploration.before_compute_actions(explore=explore, timestep=timestep)

        # assert NEIGHBOUR_ACTIONS in input_dict

        # === 处理 SampleBatch.obs 成 list ===
        #  shape: torch.Size(num_agents, 91) -> [ torch.Size(1, 91) x num_agents ]
        obs_chunk_list = torch.split(input_dict[SampleBatch.OBS], 1) 

        # === 处理 SampleBatch.infos 成 list ===
        nei_info_list = [] # [ {}, {}, ... x num_agents]
        for info in infos:
            # NOTE ppo_torch_policy._initialize_loss_from_dummy_batch()'s info contains tensor(0.)
            # e.g. (32, ): [0.0, ... x 32]
            if isinstance(info, torch.Tensor) and info == torch.tensor(0.0):
                nei_info_list.append({})
            elif isinstance(info, dict):
                nei_info_list.append(info)
            else:
                raise NotImplementedError


        if len(obs_chunk_list) != len(nei_info_list):
            printPanel(f'len(obs_chunk_list) {len(obs_chunk_list)} != len(nei_info_list) {len(nei_info_list)}', title='[Warning] action_sampler_fn()')

        obs_info_list = []
        for obs, info in zip(obs_chunk_list, nei_info_list):
            obs_info_list.append((obs, info))

        
        _out_msg = {}
        _out_msg['unsorted_obs_info_list.len'] = len(obs_info_list)

        # === 生成random_order X ===
        _order_msg = {}
        _order_msg['unsorted'] = [ (t[1].get('agent_id', 'agent0'), len(t[1].get('neighbours', []))) for t in obs_info_list]

        # 是否打乱
        if self.config['random_order']:
            random.shuffle(obs_info_list)

        # 按照邻居数量的多少排序
        if self.config['edge_descending'] != None:
            obs_info_list.sort(key=lambda x: len(x[1].get('neighbours', [])), reverse=self.config['edge_descending'])

        _order_msg['sorted'] = [(t[1].get('agent_id', 'agent0'), len(t[1].get('neighbours', []))) for t in obs_info_list]
        # printPanel(_order_msg, title=f'{self.__class__.__name__} generate random_order done!')


        # === sample actions ===
        # all_agent_actions_dict = defaultdict(lambda: torch.zeros((self.model.action_space.shape[0]))) # {agent_id: tensor.shape(2)} # for res_actions

        all_agent_actions_dict = {} # {agent_id: action}


        all_agents_nei_actions_dict = {} # for self.extra_action_out()
        all_agents_nei_actions_mask = {} # for self.extra_action_out()
        all_agent_logps_dict = {} # for return
        all_agent_dist_inputs_dict = {} # for return
        # all_agent_logps = {}
        # dist_inputs = []
        cnt = 0
        for obs, info in obs_info_list:
            agent_id = info.get('agent_id', f'agent{cnt}')
            _msg = {}
            _msg['cnt'] = cnt
            _msg['agent_id'] = agent_id
            _msg['o.shape'] = obs.shape
            _msg['info'] = info
            # printPanel(_msg, f'{self.__class__.__name__}before compting single actions')
            cnt += 1

            nei_actions = torch.zeros((1, num_neighbours, self.model.action_space.shape[0])) # shape(num_nei, 2)
            execution_mask = torch.zeros((1, num_neighbours,)) # TODO

            for i, nei_id in zip(range(num_neighbours), info.get('neighbours', [])):
                # 1. nei_action already generated
                if nei_id in all_agent_actions_dict:
                    nei_actions[0, i] = all_agent_actions_dict[nei_id]
                    execution_mask[0, i] = torch.tensor(1)
                else:
                    nei_actions[0, i] = torch.zeros((1, self.model.action_space.shape[0]))

            # nei_actions_flat = torch.reshape(nei_actions, (1,-1)) # shape(1, num_nei x 2)
            all_agents_nei_actions_dict[agent_id] = nei_actions
            all_agents_nei_actions_mask[agent_id] = execution_mask

            _msg['nei_actions.shape'] = nei_actions.shape
            _msg['nei_actions'] = nei_actions
            # o_cat = torch.cat((o, nei_actions_flat), -1) # shape (1, 91 + action_dim*num_neighbours)

            # msg["o_cat"] = o_cat.shape

            model_input_dict = {
                SampleBatch.OBS: obs,
                NEIGHBOUR_ACTIONS: nei_actions,
                EXECUTION_MASK: execution_mask,
            }
            dist_input, state_out = self.model(model_input_dict, state_batches)
            # dist_inputs.append(dist_input)
            all_agent_dist_inputs_dict[agent_id] = dist_input
            action_dist = dist_class(dist_input, self.model)
            # Get the exploration action from the forward results.
            action, logp = self.exploration.get_exploration_action(
                action_distribution=action_dist, timestep=timestep, explore=explore
            )
            all_agent_actions_dict[agent_id] = action
            all_agent_logps_dict[agent_id] = logp
            # all_agent_logps[agent_id] = logp

            # actions.append(action)
            # logps.append(logp)

            _msg['sampled_action'] = action
            _msg['logp'] = logp
            _msg['all_agent_actions_dict'] = all_agent_actions_dict
            # printPanel(_msg, 'compute single action done!')

        # inspect(input_dict[SampleBatch.OBS])


        res_actions = []
        all_agents_nei_actions_list = []
        all_agents_nei_actions_mask_list = []
        logps = []
        dist_inputs = []
        for info in nei_info_list:
            agent_id = info.get('agent_id', 'agent0')
            res_actions.append(all_agent_actions_dict[agent_id])
            all_agents_nei_actions_list.append(all_agents_nei_actions_dict[agent_id])
            all_agents_nei_actions_mask_list.append(all_agents_nei_actions_mask[agent_id])
            logps.append(all_agent_logps_dict[agent_id])
            dist_inputs.append(all_agent_dist_inputs_dict[agent_id])

        res_actions = torch.cat(res_actions, 0)
        logps = torch.cat(logps, 0)
        dist_inputs = torch.cat(dist_inputs, 0)
        self.last_computed_neighbour_actions = torch.cat(all_agents_nei_actions_list, 0)
        self.last_computed_execution_mask = torch.cat(all_agents_nei_actions_mask_list, 0)

        # msg = {}
        _out_msg['actions.shape'] = res_actions.shape
        _out_msg['actions'] = res_actions
        _out_msg['logps.shape'] = logps.shape
        # msg['logps'] = logps
        _out_msg['dist_inputs.shape'] = dist_inputs.shape
        # msg['dist_inputs'] = dist_inputs
        _out_msg['last_computed_neighbour_actions.shape'] = self.last_computed_neighbour_actions.shape
        _out_msg['last_computed_neighbour_actions'] = self.last_computed_neighbour_actions
        # printPanel(_out_msg, title='model sample actions done!')
        # print('=*= ' * 20)

        return res_actions, logps, dist_inputs, state_batches


    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        msg = {}
        msg['sample_batch'] = sample_batch
        msg['other_agent_batches'] = other_agent_batches
        msg['episode'] = episode
        # printPanel(msg, f'{self.__class__.__name__}.postprocess_trajectory()')
        # print('sample_batch', sample_batch)
        # print('other_agent_batches', other_agent_batches)
        with torch.no_grad():
            o = sample_batch[SampleBatch.CUR_OBS]
            odim = o.shape[1]

            if episode is None:
                # In initialization, we set centralized_critic_obs_dim
                if CENTRALIZED_CRITIC_OBS in sample_batch: # RLlib会第一次返回dummy_batch
                    self.centralized_critic_obs_dim = sample_batch[CENTRALIZED_CRITIC_OBS].shape[1]
                else:
                    self.centralized_critic_obs_dim = get_centralized_critic_obs_dim(self.observation_space, self.action_space, self.config.get('counterfactual'), self.config.get('num_neighbours'), self.config.get('fuse_mode'))
                    
            else:
                # After initialization, fill centralized obs
                sample_batch[CENTRALIZED_CRITIC_OBS] = np.zeros(
                    (o.shape[0], self.centralized_critic_obs_dim), dtype=sample_batch[SampleBatch.CUR_OBS].dtype
                )
                sample_batch[CENTRALIZED_CRITIC_OBS][:, :odim] = o

                assert other_agent_batches is not None
                other_info_dim = odim
                adim = sample_batch[SampleBatch.ACTIONS].shape[1]
                if self.config[COUNTERFACTUAL]:
                    other_info_dim += adim

                if self.config["fuse_mode"] == "concat":
                    sample_batch = concat_ccppo_process(
                        self, sample_batch, other_agent_batches, odim, adim, other_info_dim
                    )
                elif self.config["fuse_mode"] == "mf":
                    sample_batch = mean_field_ccppo_process(
                        self, sample_batch, other_agent_batches, odim, adim, other_info_dim
                    )
                elif self.config["fuse_mode"] == "none":
                    # Do nothing since OBS is already filled
                    assert odim == sample_batch[CENTRALIZED_CRITIC_OBS].shape[1]
                else:
                    raise ValueError("Unknown fuse mode: {}".format(self.config["fuse_mode"]))

            # Use centralized critic to compute the value
            sample_batch[SampleBatch.VF_PREDS] = self.model.central_value_function(
                convert_to_torch_tensor(sample_batch[CENTRALIZED_CRITIC_OBS], self.device)
            ).cpu().detach().numpy().astype(np.float32)

            if sample_batch[SampleBatch.DONES][-1]:
                last_r = 0.0
            else:
                last_r = sample_batch[SampleBatch.VF_PREDS][-1]
            batch = compute_advantages(
                sample_batch,
                last_r,
                self.config["gamma"],
                self.config["lambda"],
                use_gae=self.config["use_gae"],
                use_critic=self.config.get("use_critic", True)
            )
        return batch

    def loss(self, model, dist_class, train_batch):
        """
        Compute loss for Proximal Policy Objective.

        PZH: We replace the value function here so that we query the centralized values instead
        of the native value function.
        """
        _msg = {}
        _msg['train_batch.is_training'] = train_batch.is_training
        # printPanel(_msg, f'{self.__class__.__name__}.loss()')

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES] *
            torch.clamp(logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]),
        )

        # Compute a value function loss.
        assert self.config["use_critic"]

        # ========== Modification ==========
        # value_fn_out = model.value_function()
        value_fn_out = self.model.central_value_function(train_batch[CENTRALIZED_CRITIC_OBS])
        # ========== Modification Ends ==========

        if self.config["old_value_loss"]:
            current_vf = value_fn_out
            prev_vf = train_batch[SampleBatch.VF_PREDS]
            vf_loss1 = torch.pow(current_vf - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_clipped = prev_vf + torch.clamp(
                current_vf - prev_vf, -self.config["vf_clip_param"], self.config["vf_clip_param"]
            )
            vf_loss2 = torch.pow(vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_loss_clipped = torch.max(vf_loss1, vf_loss2)
        else:
            vf_loss = torch.pow(value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)

        total_loss = reduce_mean_valid(
            -surrogate_loss + self.config["vf_loss_coeff"] * vf_loss_clipped - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss


class ARCCPPOTrainer(IPPOTrainer):
    @classmethod
    def get_default_config(cls):
        return ARCCPPOConfig()

    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        assert config["framework"] == "torch"
        return ARCCPPOPolicy

