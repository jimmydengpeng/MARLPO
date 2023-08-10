from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.sgd import standardized
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from .algo_ippo import IPPOConfig, IPPOPolicy, IPPOTrainer
from utils.debug import get_logger
logger = get_logger()

torch, nn = try_import_torch()


class IPPORSConfig(IPPOConfig):
    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class=algo_class or IPPORSTrainer)
        # Two important updates


class IPPORSPolicy(IPPOPolicy):

    @override(IPPOPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        '''Args:
            sample_batch: Dict[agent_id, Tuple['default_policy', Policy, SampleBatch]]
        '''
        msg = {}
        msg['sample_batch'] = sample_batch

        with torch.no_grad():
            sample_batch[ORIGINAL_REWARDS] = sample_batch[SampleBatch.REWARDS].copy()
            sample_batch[NEI_REWARDS] = sample_batch[SampleBatch.REWARDS].copy() # for init ppo
            # sample_batch[TEAM_REWARDS] = sample_batch[SampleBatch.REWARDS].copy() # for init ppo
            # sample_batch[NEI_VALUES] = self.model.get_nei_value().cpu().detach().numpy().astype(np.float32)

            if episode: # filter _initialize_loss_from_dummy_batch()
                msg['*'] = '*'
                msg['agent id'] = f'agent{sample_batch[SampleBatch.AGENT_INDEX][0]}'
                # msg['agent id last'] = f'agent{sample_batch[SampleBatch.AGENT_INDEX][-1]}'
                # print('*'*30)
                # print('t:', sample_batch['t'][:3])
                # print('agent_index:', sample_batch['agent_index'][:3])
                # print('actions:', sample_batch['actions'][:3])
                # print('rewards:', sample_batch['rewards'][:3])
                # print('nei rewards:', sample_batch[NEI_REWARDS][:3])
                # print('infos:', sample_batch['infos'][:3])
                # print('#'*30)
                # == 1. add neighbour rewards ==
                sample_batch = add_neighbour_rewards(self.config, sample_batch)

                # == 2. compute team rewards ==
                nei_r_coeff = self.config.get('nei_rewards_add_coeff', 1)
                sample_batch[TEAM_REWARDS] = sample_batch[ORIGINAL_REWARDS] + nei_r_coeff * sample_batch[NEI_REWARDS] 

            if sample_batch[SampleBatch.DONES][-1]:
                last_r = last_nei_r = last_team_r = 0.0
            else:
                last_r = sample_batch[SampleBatch.VF_PREDS][-1]
                # last_nei_r = sample_batch[NEI_VALUES][-1]
                last_team_r = sample_batch[TEAM_VALUES][-1]

            
            compute_advantages(
                sample_batch,
                last_r,
                self.config["gamma"],
                self.config["lambda"],
                use_gae=self.config["use_gae"],
                use_critic=self.config.get("use_critic", True)
            )

            sample_batch = _compute_advantage(
                sample_batch, 
                (TEAM_REWARDS, TEAM_VALUES, TEAM_ADVANTAGES, TEAM_VALUE_TARGETS),
                last_team_r, 
                self.config["gamma"], 
                self.config["lambda"],
            )

            # == 记录 rollout 时的 V 值 == 
            # vpred_t = np.concatenate([sample_batch[SampleBatch.VF_PREDS], np.array([last_r])])
            # sample_batch[NEXT_VF_PREDS] = vpred_t[1:].copy()

        return sample_batch


    def loss(self, model, dist_class, train_batch):

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

        # == normalize advatages ==
        if self.config.get('norm_adv', False):
            adv = standardized(train_batch[Postprocessing.ADVANTAGES])
            logger.warning(
                "train_batch[Postprocessing.ADVANTAGES].mean(): {}".format(
                    torch.mean(train_batch[Postprocessing.ADVANTAGES])
                )
            )
        else:
            adv = train_batch[Postprocessing.ADVANTAGES]

        surrogate_loss = torch.min(
            adv * logp_ratio,
            adv *
            torch.clamp(logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]),
        )

        # Compute a value function loss.
        assert self.config["use_critic"]

        value_fn_out = model.value_function()

        # === 使用IPPO中的 Value Loss
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


class IPPORSTrainer(IPPOTrainer):
    @classmethod
    def get_default_config(cls):
        return IPPORSConfig()

    def get_default_policy_class(self, config):
        assert config["framework"] == "torch"
        return IPPORSPolicy



__all__ = ['IPPORSConfig', 'IPPORSTrainer']