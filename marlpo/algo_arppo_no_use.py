from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)


import copy
import functools
import logging
import math
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree

import ray
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import _directStepOptimizerSingleton
from ray.rllib.utils import NullContextManager, force_list
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.annotations import (
    DeveloperAPI,
    OverrideToImplementCustomLogic,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
    is_overridden,
    override,
)
from ray.rllib.utils.error import ERR_MSG_TORCH_POLICY_CANNOT_SAVE_MODEL
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY,
    NUM_AGENT_STEPS_TRAINED,
    NUM_GRAD_UPDATES_LIFETIME,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    GradInfoDict,
    ModelGradients,
    ModelWeights,
    PolicyState,
    TensorStructType,
    TensorType,
)


from marlpo.algo_ippo import IPPOTrainer, IPPOConfig

if TYPE_CHECKING:
    from ray.rllib.evaluation import Episode  # noqa

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

from rich import inspect





class ARPPOPolicy(PPOTorchPolicy):

    def __init__(self, observation_space, action_space, config):
        self.count = 0
        super().__init__(observation_space, action_space, config)
        print('>>> [ARPPOPolicy] init()...')

    def loss(self, model, dist_class, train_batch):
        """
        Compute loss for Proximal Policy Objective.

        PZH: We replace the value function here so that we query the centralized values instead
        of the native value function.
        """

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

        value_fn_out = model.value_function()

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

    @override(Policy)
    def compute_actions_from_input_dict(
        self,
        input_dict: Dict[str, TensorType],
        explore: bool = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        self.count += 1
        print('>>> [ARPPOPolicy] compute_actions_from_input_dict()... {}'.format(self.count))

        with torch.no_grad():
            
            # for k in input_dict:
            #     print(k, ':', input_dict[k])
            # inspect(input_dict)

            # Pass lazy (torch) tensor dict to Model as `input_dict`.
            input_dict = self._lazy_tensor_dict(input_dict)
            input_dict.set_training(True)
            # Pack internal state inputs into (separate) list.
            state_batches = [
                input_dict[k] for k in input_dict.keys() if "state_in" in k[:8]
            ]
            # Calculate RNN sequence lengths.
            seq_lens = (
                torch.tensor(
                    [1] * len(state_batches[0]),
                    dtype=torch.long,
                    device=state_batches[0].device,
                )
                if state_batches
                else None
            )

            # obs = input_dict[SampleBatch.OBS]
            # inspect(input_dict)
            # print(input_dict[SampleBatch.AGENT_INDEX])
            # print(obs.shape)
            # print(input_dict.keys())
            # print(type(input_dict))
            # print(input_dict.keys())
            # print(type(input_dict['t']))

            return self._compute_action_helper(
                input_dict, state_batches, seq_lens, explore, timestep
            )


    @with_lock
    def _compute_action_helper(
        self, input_dict, state_batches, seq_lens, explore, timestep
    ):
        """Shared forward pass logic (w/ and w/o trajectory view API).

        Returns:
            A tuple consisting of 
                a) actions, -> torch.Size([n_agents, act_dim])
                b) state_out, -> []
                c) extra_fetches. -> {'vf_preds': tensor, 'action_dist_inputs': ...}
            The input_dict is modified in-place to include a numpy copy of the computed
            actions under `SampleBatch.ACTIONS`.
        """
        print('>>> [ARPPOPolicy] _compute_action_helper()...')

        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep
        self._is_recurrent = state_batches is not None and state_batches != []

        # Switch to eval mode.
        if self.model:
            self.model.eval()

        extra_fetches = {}
        if isinstance(self.model, RLModule):
            if explore:
                fwd_out = self.model.forward_exploration(input_dict)
            else:
                fwd_out = self.model.forward_inference(input_dict)
            # anything but action_dist and state_out is an extra fetch
            action_dist = fwd_out.pop("action_dist")

            if explore:
                actions, logp = action_dist.sample(return_logp=True)
            else:
                actions = action_dist.sample()
                logp = None
            state_out = fwd_out.pop("state_out", {})
            extra_fetches = fwd_out
            dist_inputs = None
        elif is_overridden(self.action_sampler_fn):
            action_dist = dist_inputs = None
            actions, logp, state_out = self.action_sampler_fn(
                self.model,
                obs_batch=input_dict,
                state_batches=state_batches,
                explore=explore,
                timestep=timestep,
            )
        else:
            use_ar_policy = False
            # Call the exploration before_compute_actions hook.
            self.exploration.before_compute_actions(explore=explore, timestep=timestep)
            if is_overridden(self.action_distribution_fn):
                dist_inputs, dist_class, state_out = self.action_distribution_fn(
                    self.model,
                    obs_batch=input_dict,
                    state_batches=state_batches,
                    seq_lens=seq_lens,
                    explore=explore,
                    timestep=timestep,
                    is_training=False,
                )
            else:
                use_ar_policy = True
                dist_class = self.dist_class
                
                dist_inputs, state_out = self.model(input_dict, state_batches, seq_lens)

                # === Implement the Auto-Regressive Policy here! ===

                # input_dict.keys(): ['obs', 'new_obs', 'actions', 'prev_actions', 'rewards', 'prev_rewards', 'terminateds', 'truncateds', 'infos', 'eps_id', 'unroll_id', 'agent_index', 't']) -> firts 32 test?
                # sample时只有'obs'可用

                # print(input_dict.keys())
                assert state_batches == []
                assert seq_lens == None

                single_agent_input_dict = input_dict.copy()
                obs = single_agent_input_dict[SampleBatch.OBS] # size(n_agents, obs_dim)
                obs_in_chunk = torch.split(obs, 1)

                dist_inputs = []
                action_dist = []
                actions = []
                logp = []
                for o in obs_in_chunk:
                    # o = torch.unsqueeze(o, 0)
                    # print(o.shape)
                    single_agent_input_dict[SampleBatch.OBS] = o

                    dist_input, state_out = self.model(single_agent_input_dict, state_batches, seq_lens)
                    dist_inputs.append(dist_input)
                    assert state_out == []


                    _action_dist = dist_class(dist_input, self.model)
                    action_dist.append(_action_dist)

                    # Get the exploration action from the forward results.
                    action, _logp = self.exploration.get_exploration_action(
                        action_distribution=_action_dist, timestep=timestep, explore=explore
                    )
                    actions.append(action)
                    logp.append(_logp)
                
                dist_inputs = torch.squeeze(torch.stack(dist_inputs), 1)
                # action_dist = torch.squeeze(torch.stack(action_dist), 1)
                actions = torch.squeeze(torch.stack(actions), 1)
                logp = torch.squeeze(torch.stack(logp), 1)
                print(actions.shape)
                print(logp.shape)
                # actions = np.array(actions)
                # logp = np.array(logp_)



            if not (
                isinstance(dist_class, functools.partial)
                or issubclass(dist_class, TorchDistributionWrapper)
            ):
                raise ValueError(
                    "`dist_class` ({}) not a TorchDistributionWrapper "
                    "subclass! Make sure your `action_distribution_fn` or "
                    "`make_model_and_action_dist` return a correct "
                    "distribution class.".format(dist_class.__name__)
                )
            
            # 如果不适用AR-Policy
            if not use_ar_policy:
                action_dist = dist_class(dist_inputs, self.model)

                # Get the exploration action from the forward results.
                actions, logp = self.exploration.get_exploration_action(
                    action_distribution=action_dist, timestep=timestep, explore=explore
                ) # size(n_agents, act_dim), size(n_agents)

        # Add default and custom fetches.
        # 计算Value_function
        if not extra_fetches:
            # input_dict: SampleBatch(n_agents: ['obs'])
            # state_batches: []
            # action_dist: TorchDiagGaussian
            print('>>>>>>')
            print(input_dict)
            print(state_batches)
            print(action_dist)
            extra_fetches = self.extra_action_out(
                input_dict, state_batches, self.model, action_dist # action_dist 未使用？
            )

        # Action-dist inputs.
        if dist_inputs is not None:
            extra_fetches[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs

        # Action-logp and action-prob.
        if logp is not None:
            extra_fetches[SampleBatch.ACTION_PROB] = torch.exp(logp.float())
            extra_fetches[SampleBatch.ACTION_LOGP] = logp

        # Update our global timestep by the batch size.
        self.global_timestep += len(input_dict[SampleBatch.CUR_OBS])
        return convert_to_numpy((actions, state_out, extra_fetches))



class ARPPOTrainer(PPO):
    @classmethod
    def get_default_config(cls):
        return ARPPOConfig()

    def get_default_policy_class(self, config):
        assert config["framework"] == "torch"
        return ARPPOPolicy




CENTRALIZED_CRITIC_OBS = "centralized_critic_obs"
COUNTERFACTUAL = "counterfactual"


class ARPPOConfig(IPPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or ARPPOTrainer)
        self.counterfactual = True
        self.num_neighbours = 4
        self.fuse_mode = "mf"  # In ["concat", "mf", "none"]
        self.mf_nei_distance = 10
        self.old_value_loss = True
        self.update_from_dict({"model": {"custom_model": "cc_model"}})

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


class ARModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""
    def __init__(
        self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
        model_config: ModelConfigDict, name: str, **model_kwargs
    ):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **model_kwargs)
        nn.Module.__init__(self)

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(model_config.get("post_fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")
        # Generate free-floating bias variables for the second half of
        # the outputs.
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two",
                num_outputs,
            )
            num_outputs = num_outputs // 2

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # ========== Our Modification: We compute the centralized critic obs size here! ==========
        centralized_critic_obs_dim = self.get_centralized_critic_obs_dim()

        # TODO === Fix WARNING catalog.py:617 -- Custom ModelV2 should accept all custom options as **kwargs, instead of expecting them in config['custom_model_config']! === 
        self.fuse_mode = model_kwargs.get("fuse_mode", None)
        self.counterfactual = model_kwargs.get("counterfactual", False)
        self.num_neighbours = model_kwargs.get("num_neighbours", 0)


        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation
                )
            )
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation
                )
            )
            prev_layer_size = num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=hiddens[-1],
                        initializer=normc_initializer(1.0),
                        activation_fn=activation
                    )
                )
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(0.01),
                    activation_fn=None
                )
            else:
                self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[-1]

        # Layer to add the log std vars to the state-dependent means.
        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.

            # ========== Our Modification ==========
            # Note: We use centralized critic obs size as the input size of critic!
            # prev_vf_layer_size = int(np.product(obs_space.shape))
            prev_vf_layer_size = centralized_critic_obs_dim
            assert prev_vf_layer_size > 0

            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0)
                    )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
            in_size=prev_vf_layer_size, out_size=1, initializer=normc_initializer(0.01), activation_fn=None
        )

        self.view_requirements[CENTRALIZED_CRITIC_OBS] = ViewRequirement(
            space=Box(obs_space.low[0], obs_space.high[0], shape=(centralized_critic_obs_dim, ))
        )

        self.view_requirements[SampleBatch.ACTIONS] = ViewRequirement(space=action_space)

    def get_centralized_critic_obs_dim(self):
        return get_centralized_critic_obs_dim(
            self.obs_space, self.action_space, self.model_config["custom_model_config"]["counterfactual"],
            self.model_config["custom_model_config"]["num_neighbours"],
            self.model_config["custom_model_config"]["fuse_mode"]
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        obs = obs.reshape(obs.shape[0], -1)
        features = self._hidden_layers(obs)
        logits = self._logits(features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        raise ValueError(
            "Centralized Value Function should not be called directly! "
            "Call central_value_function(cobs) instead!"
        )

    def central_value_function(self, obs):
        assert self._value_branch is not None
        return torch.reshape(self._value_branch(self._value_branch_separate(obs)), [-1])


ModelCatalog.register_custom_model("cc_model", ARModel)


def concat_ccppo_process(policy, sample_batch, other_agent_batches, odim, adim, other_info_dim):
    """Concat the neighbors' observations"""
    for index in range(sample_batch.count):

        environmental_time_step = sample_batch["t"][index]

        # "neighbours" may not be in sample_batch['infos'][index]:
        # neighbours = sample_batch['infos'][index]["neighbours"]
        neighbours = sample_batch['infos'][index].get("neighbours", [])

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


def get_ccppo_env(env_class):
    return get_rllib_compatible_new_gymnasium_api_env(get_ccenv(env_class))


class CCPPOPolicy(PPOTorchPolicy):
    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        return {}

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
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


class ARPPOTrainer(IPPOTrainer):
    @classmethod
    def get_default_config(cls):
        return ARPPOConfig()

    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        assert config["framework"] == "torch"
        return CCPPOPolicy


