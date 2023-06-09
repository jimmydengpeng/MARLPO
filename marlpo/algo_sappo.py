import gym
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import (
    Postprocessing, 
    compute_gae_for_sample_batch, 
    compute_advantages
    )
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import (
    DeveloperAPI,
    OverrideToImplementCustomLogic,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
    is_overridden,
    override,
)
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)

from marlpo.models.sa_model import SAModel
from marlpo.utils import inspect, printPanel

torch, nn = try_import_torch()

ModelCatalog.register_custom_model("sa_model", SAModel)


class SAPPOConfig(PPOConfig):
    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class=algo_class or SAPPOTrainer)
        # IPPO params
        self.vf_clip_param = 100
        self.old_value_loss = True

        # common
        self.num_neighbours = 4

        # Central Critic
        self.use_central_critic = False
        self.counterfactual = True
        self.fuse_mode = "mf"  # In ["concat", "mf", "none"]
        self.mf_nei_distance = 10

        # Attention Encoder
        self.use_attention = True
        
        # Custom Model configs
        self.update_from_dict({"model": {"custom_model": "sa_model"}})


    def validate(self):
        super().validate()
        assert self["fuse_mode"] in ["mf", "concat", "none"]

        # common
        self.model["custom_model_config"]["num_neighbours"] = self["num_neighbours"]

        # Central Critic
        self.model["custom_model_config"]["use_central_critic"] = self["use_central_critic"]
        self.model["custom_model_config"]["fuse_mode"] = self["fuse_mode"]
        self.model["custom_model_config"]["counterfactual"] = self["counterfactual"]

        # Attention Encoder
        self.model["custom_model_config"]["use_attention"] = self["use_attention"]

        # get obs_shape for every env_config for attention encoder
        self.model["custom_model_config"]["env_config"] = self.env_config


        # from ray.tune.registry import get_trainable_cls
        # env_cls = get_trainable_cls(self.env)
        # printPanel({'env_cls': env_cls}, color='red')
        obs_shape = self.model["custom_model_config"]['env_cls'].get_obs_shape(self.env_config)
        msg = {}
        msg['obs_shape'] = obs_shape
        msg['custom_model_config'] = self.model["custom_model_config"]
        # printPanel(msg, title='SAPPOConfig validate', color='green')


class SAPPOPolicy(PPOTorchPolicy):
    # @override(PPOTorchPolicy)
    # def extra_action_out(self, input_dict, state_batches, model, action_dist):
    #     return {
    #         "SVO": model.last_svo,
    #         SampleBatch.VF_PREDS: model.value_function(),
    #     }


    # @override(PPOTorchPolicy)
    # def postprocess_trajectory(
    #     self, sample_batch, other_agent_batches=None, episode=None
    # ):
    #     msg = {}
    #     msg['sample_batch'] = sample_batch
    #     msg['other_agent_batches'] = other_agent_batches
    #     msg['episode'] = episode
    #     printPanel(msg, f'{self.__class__.__name__}.postprocess_trajectory()')
    #     # print('sample_batch', sample_batch)
    #     # print('other_agent_batches', other_agent_batches)
    #     with torch.no_grad():
    #         o = sample_batch[SampleBatch.CUR_OBS]
    #         odim = o.shape[1]

    # #         # if sample_batch[SampleBatch.DONES][-1]:
    # #         #     last_r = 0.0
    # #         # else:
    # #         #     last_r = sample_batch[SampleBatch.VF_PREDS][-1]
           
    # #         # batch = compute_advantages(
    # #         #     sample_batch,
    # #         #     last_r,
    # #         #     self.config["gamma"],
    # #         #     self.config["lambda"],
    # #         #     use_gae=self.config["use_gae"],
    # #         #     use_critic=self.config.get("use_critic", True)
    # #         # )

    #         return compute_gae_for_sample_batch(
    #             self, sample_batch, other_agent_batches, episode
    #         )
    # #     return batch



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

        # === 使用IPPO中的 Value Loss ===
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


class SAPPOTrainer(PPO):
    @classmethod
    def get_default_config(cls):
        return SAPPOConfig()

    def get_default_policy_class(self, config):
        assert config["framework"] == "torch"
        return SAPPOPolicy

