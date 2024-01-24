import logging
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple

from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_learner import LEARNER_RESULTS_KL_KEY
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.execution.rollout_ops import standardize_fields, synchronous_parallel_sample
from ray.rllib.execution.train_ops import train_one_step, multi_gpu_train_one_step
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import ExperimentalAPI, override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
    SAMPLE_TIMER,
    ALL_MODULES,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.rllib.utils.sgd import standardized
from ray.rllib.utils.torch_utils import sequence_mask, warn_if_infinite_kl_divergence
from ray.rllib.utils.typing import Dict, ResultDict, TensorType, List, Tuple, ModelConfigDict
from ray.util.debug import log_once

from algo.algo_ippo_rs import IPPORSConfig, IPPORSPolicy, IPPORSTrainer
from algo.utils import add_neighbour_rewards, orthogonal_initializer, _compute_advantage
from utils.debug import printPanel, reduce_window_width
reduce_window_width(__file__)

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)

ORIGINAL_REWARDS = "original_rewards"
NEI_REWARDS = "nei_rewards"
NEXT_VF_PREDS = 'next_vf_preds'
HAS_NEIGHBOURS = 'has_neighbours'

NEI_REWARDS = "nei_rewards"
NEI_VALUES = "nei_values"
NEI_ADVANTAGES = "nei_advantages"
NEI_TARGETS = "nei_targets"

NEI_REWARDS_MODE = 'nei_rewards_mode'

MEAN_NEI_REWARDS = 'mean_nei_rewards'                 # ─╮ 
MAX_NEI_REWARDS = 'max_nei_rewards'                   #  │
NEAREST_NEI_REWARDS = 'nearest_nei_reward'            #  │──> Choose 1 
ATTENTIVE_ONE_NEI_REWARD = 'attentive_one_nei_reward' #  │
ATTENTIVE_ALL_NEI_REWARD = 'attentive_all_nei_reward' # ─╯


# IPPO Double-Critic
# class IPPODCConfig(IPPORSConfig):
class IPPODCConfig(PPOConfig):
    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class=algo_class or IPPODCTrainer)
        # IPPO params
        self.vf_clip_param = 100
        self.old_value_loss = True

        # common
        self.num_neighbours = 4

        self.nei_rewards_mode='mean'
        self.nei_reward_if_no_nei='self'
        self.norm_adv=True
        

        # self.model["custom_model_config"]["num_neighbours"] = self["num_neighbours"]
        # self.update_from_dict({"model": {"custom_model": "ippo_dc_model"}})

    def validate(self):
        super().validate()
        self.model["custom_model_config"]["num_neighbours"] = self["num_neighbours"]
        self.update_from_dict({"model": {"custom_model": "ippo_dc_model"}})


class IPPODCModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.custom_model_config = model_config['custom_model_config']

        hiddens = list(model_config.get("fcnet_hiddens", [])) # [256, 256]
        activation = model_config.get("fcnet_activation") # tanh by default

        # == policy ==
        in_size = int(np.product(obs_space.shape))
        self.idv_policy_network = self.build_one_policy_network(
                                    in_size=in_size,
                                    out_size=num_outputs,
                                    activation=activation,
                                    hiddens=hiddens)

        # == critics ==
        in_size = int(np.product(obs_space.shape))
        self.idv_value_network = self.build_one_value_network(
                                    in_size=in_size,
                                    activation=activation,
                                    hiddens=hiddens)
        self.nei_value_network = self.build_one_value_network(
                                    in_size=in_size,
                                    activation=activation,
                                    hiddens=hiddens)

        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

        self.view_requirements[NEI_REWARDS] = ViewRequirement()
        self.view_requirements[NEI_VALUES] = ViewRequirement()
        self.view_requirements[NEI_TARGETS] = ViewRequirement()
        self.view_requirements[NEI_ADVANTAGES] = ViewRequirement()


    def build_one_policy_network(self, in_size, out_size, activation, hiddens):
        layers = []
        prev_layer_size = in_size
        # Create layers 0 to second-last.
        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = size

        layers.append(SlimFC(
            in_size=prev_layer_size,
            out_size=out_size,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        ))
        return nn.Sequential(*layers)

    def build_one_value_network(self, in_size, activation, hiddens):
        assert in_size > 0
        vf_layers = []
        for size in hiddens:
            vf_layers.append(
                SlimFC(
                    in_size=in_size, 
                    out_size=size, 
                    activation_fn=activation, 
                    # initializer=normc_initializer(1.0)
                    initializer=orthogonal_initializer(gain=1.0)
                ))
            in_size = size
        vf_layers.append(
            SlimFC(
                in_size=in_size, 
                out_size=1, 
                # initializer=normc_initializer(0.01), 
                initializer=orthogonal_initializer(gain=0.1),
                activation_fn=None
            ))
        return nn.Sequential(*vf_layers)

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        logits = self.idv_policy_network(self._last_flat_in)
        return logits, state
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._last_flat_in is not None, "must call forward() first"
        out = self.idv_value_network(self._last_flat_in).squeeze(1)
        return out

    def nei_value_function(self):
        assert self._last_flat_in is not None, "must call forward() first"
        out = self.nei_value_network(self._last_flat_in).squeeze(1)
        return out


ModelCatalog.register_custom_model("ippo_dc_model", IPPODCModel)


class IPPODCPolicy(PPOTorchPolicy):
# class IPPODCPolicy(IPPORSPolicy):
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        # IPPORSPolicy.__init__(self, observation_space, action_space, config)

    @override(PPOTorchPolicy)
    # @override(IPPORSPolicy)
    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        return {
            SampleBatch.VF_PREDS: model.value_function(),
            NEI_VALUES: model.nei_value_function(),
        }

    
    # @override(IPPORSPolicy)
    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        with torch.no_grad():
            sample_batch[ORIGINAL_REWARDS] = sample_batch[SampleBatch.REWARDS].copy()
            sample_batch[NEI_REWARDS] = sample_batch[SampleBatch.REWARDS].copy() # for init ppo

            if episode: # filter _initialize_loss_from_dummy_batch()
                # == 1. add neighbour rewards ==
                sample_batch = add_neighbour_rewards(self.config, sample_batch)


            if sample_batch[SampleBatch.DONES][-1]:
                last_r = last_nei_r = 0.0
            else:
                last_r = sample_batch[SampleBatch.VF_PREDS][-1]
                last_nei_r = sample_batch[NEI_VALUES][-1]

            # RLlib's compute_advantages() only computes raw individual environmental rewards!
            compute_advantages(
                sample_batch,
                last_r,
                self.config["gamma"],
                self.config["lambda"],
                use_gae=self.config["use_gae"],
                use_critic=self.config.get("use_critic", True)
            )

            # compute team advantage
            sample_batch = _compute_advantage(
                sample_batch, 
                (NEI_REWARDS, NEI_VALUES, NEI_ADVANTAGES, NEI_TARGETS),
                last_nei_r, 
                self.config["gamma"], 
                self.config["lambda"],
            )

        return sample_batch


    def loss(self, model, dist_class, train_batch: SampleBatch):

        # == get logits & dists ==
        idv_logits, state = model(train_batch)
        curr_action_dist = dist_class(idv_logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = idv_logits.shape[0] // B
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

        # == compute ratios ==
        prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0: # 约束新旧策略更新, 默认为0.2 
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        # == compute advantages ==
        idv_adv = train_batch[Postprocessing.ADVANTAGES]
        nei_adv = train_batch[NEI_ADVANTAGES]
        adv = idv_adv + self.config['phi'] * nei_adv

        # == normalize advatages ==
        if self.config.get('norm_adv', False):
            adv = standardized(adv)
            logger.warning(
                "train_batch[Postprocessing.ADVANTAGES].mean(): {}".format(
                    torch.mean(train_batch[Postprocessing.ADVANTAGES])))

        # == obj ==
        surrogate_loss = torch.min(
            adv * logp_ratio,
            adv *
            torch.clamp(logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]),
        )


        # == critic loss ==
        assert self.config["use_critic"]

        if self.config["old_value_loss"]:
            # == IPPO Value Loss ==
            def _compute_value_loss(current_vf, prev_vf, value_target):
                vf_loss1 = torch.pow(current_vf - value_target, 2.0)
                vf_clipped = prev_vf + torch.clamp(
                    current_vf - prev_vf, -self.config["vf_clip_param"], self.config["vf_clip_param"]
                )
                vf_loss2 = torch.pow(vf_clipped - value_target, 2.0)
                vf_loss = torch.max(vf_loss1, vf_loss2)
                return vf_loss
        else:
            # == 使用原始 PPO 的 value loss ==
            def _compute_value_loss(current_vf, prev_vf, value_target):
                vf_loss = torch.pow(current_vf - value_target, 2.0)
                vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
                return vf_loss_clipped

        idv_vf_loss = _compute_value_loss(
            current_vf=model.value_function(),
            prev_vf=train_batch[SampleBatch.VF_PREDS],
            value_target=train_batch[Postprocessing.VALUE_TARGETS]
        )

        nei_vf_loss = _compute_value_loss(
            current_vf=model.nei_value_function(),
            prev_vf=train_batch[NEI_VALUES],
            value_target=train_batch[NEI_TARGETS]
        )


        # == total loss ==
        total_loss = reduce_mean_valid(
            - surrogate_loss \
            + self.config["vf_loss_coeff"] * idv_vf_loss \
            + self.config["vf_loss_coeff"] * nei_vf_loss \
            - self.entropy_coeff * curr_entropy # self.entropy_coeff=0 by default
        )
        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss       


        # === STATS ===
        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = reduce_mean_valid(idv_vf_loss)
        model.tower_stats["mean_idv_vf_loss"] = reduce_mean_valid(idv_vf_loss)
        model.tower_stats["mean_nei_vf_loss"] = reduce_mean_valid(nei_vf_loss)
        
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss


    @override(PPOTorchPolicy)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_idv_vf_loss"))
                ),
                "idv_vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_idv_vf_loss"))
                ),
                "nei_vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_nei_vf_loss"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
            }
        )


# class IPPODCTrainer(IPPORSTrainer):
class IPPODCTrainer(PPO):
    @classmethod
    def get_default_config(cls):
        return IPPODCConfig()

    def get_default_policy_class(self, config):
        assert config["framework"] == "torch"
        return IPPODCPolicy

