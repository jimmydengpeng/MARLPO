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

from algo.utils import add_neighbour_rewards, orthogonal_initializer, _compute_advantage
from utils.debug import printPanel, reduce_window_width
reduce_window_width(__file__)

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)

ORIGINAL_REWARDS = "original_rewards"
NEI_REWARDS = "nei_rewards"
SVO = 'svo'
NEXT_VF_PREDS = 'next_vf_preds'
HAS_NEIGHBOURS = 'has_neighbours'
ATTENTION_MAXTRIX = 'attention_maxtrix'

NEI_REWARDS = "nei_rewards"
NEI_VALUES = "nei_values"
NEI_ADVANTAGE = "nei_advantage"
NEI_TARGET = "nei_target"

TEAM_REWARDS = "team_rewards"
TEAM_VALUES = "team_values"
TEAM_ADVANTAGES = "team_advantages"
TEAM_VALUE_TARGETS = "team_target"

NEI_REWARDS_MODE = 'nei_rewards_mode'

MEAN_NEI_REWARDS = 'mean_nei_rewards'                 # ─╮ 
MAX_NEI_REWARDS = 'max_nei_rewards'                   #  │
NEAREST_NEI_REWARDS = 'nearest_nei_reward'            #  │──> Choose 1 
ATTENTIVE_ONE_NEI_REWARD = 'attentive_one_nei_reward' #  │
ATTENTIVE_ALL_NEI_REWARD = 'attentive_all_nei_reward' # ─╯


class SCPOConfig(PPOConfig):
    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class=algo_class or SCPOTrainer)
        # IPPO params
        self.vf_clip_param = 100
        self.old_value_loss = True

        # common
        self.num_neighbours = 4

        self.nei_rewards_mode='mean'
        self.nei_reward_if_no_nei='self'
        self.norm_adv=True
        self.idv_policy_no_team=True,

    def validate(self):
        super().validate()
        self.model["custom_model_config"]["num_neighbours"] = self["num_neighbours"]
        self.update_from_dict({"model": {"custom_model": "scpo_model"}})


class SCPOModel(TorchModelV2, nn.Module):
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

        # == policies ==
        in_size = int(np.product(obs_space.shape))
        self.ind_policy_network = self.build_one_policy_network(
                                    in_size=in_size,
                                    out_size=num_outputs,
                                    activation=activation,
                                    hiddens=hiddens)
        self.team_policy_network = self.build_one_policy_network(
                                    in_size=in_size,
                                    out_size=num_outputs,
                                    activation=activation,
                                    hiddens=hiddens)
        # == critics ==
        in_size = int(np.product(obs_space.shape))
        self.ind_value_network = self.build_one_value_network(
                                    in_size=in_size,
                                    activation=activation,
                                    hiddens=hiddens)
        self.team_value_network = self.build_one_value_network(
                                    in_size=in_size,
                                    activation=activation,
                                    hiddens=hiddens)

        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

        # self.view_requirements[SVO] = ViewRequirement()
        # self.view_requirements[ORIGINAL_REWARDS] = ViewRequirement(data_col=SampleBatch.REWARDS, shift=0)
        # self.view_requirements[HAS_NEIGHBOURS] = ViewRequirement()

        self.view_requirements[NEI_REWARDS] = ViewRequirement()
        self.view_requirements[NEI_VALUES] = ViewRequirement()
        self.view_requirements[NEI_TARGET] = ViewRequirement()
        self.view_requirements[NEI_ADVANTAGE] = ViewRequirement()

        self.view_requirements[TEAM_REWARDS] = ViewRequirement()
        self.view_requirements[TEAM_VALUES] = ViewRequirement()
        self.view_requirements[TEAM_VALUE_TARGETS] = ViewRequirement()
        self.view_requirements[TEAM_ADVANTAGES] = ViewRequirement()

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
        logits = self.ind_policy_network(self._last_flat_in)
        return logits, state
    
    def team_policy(self) -> Tuple[TensorType, List[TensorType]]:
        assert self._last_flat_in is not None, "must call forward() first"
        logits = self.team_policy_network(self._last_flat_in)
        return logits

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._last_flat_in is not None, "must call forward() first"
        out = self.ind_value_network(self._last_flat_in).squeeze(1)
        return out

    def team_value_function(self):
        assert self._last_flat_in is not None, "must call forward() first"
        out = self.team_value_network(self._last_flat_in).squeeze(1)
        return out

    def check_params_updated(self, name: str):

        def check_parameters_updated(model: nn.modules):
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    return True, param.grad
            return False, None

        assert name in ['svo', 'nei_critic']
        if name == 'svo':
            layers = getattr(self, f'_svo_layer')
        elif name == 'nei_critic':
            layers = getattr(self, f'nei_value_network')

        res, grad = check_parameters_updated(layers)

        # printPanel(
        #     {'params updated': res,
        #      'params slice': list(layers.parameters())[-2][-1][-5:],
        #      'params grad': grad,
        #      'grad std/mean': None if grad == None else torch.std_mean(grad),
        #     }, 
        #     title=f'check_{name}_params'
        # )


ModelCatalog.register_custom_model("scpo_model", SCPOModel)




# TODO: more precise
class SCPOKLCoeffSchedule:
    def __init__(self, config):
        idv_schedule = config['idv_kl_coeff_schedule'] 
        team_schedule = config['team_kl_coeff_schedule'] 
        self.idv_kl_coeff = idv_schedule[0][-1]
        self.team_kl_coeff = team_schedule[0][-1]

        self._idv_entropy_coeff_schedule = PiecewiseSchedule(
            endpoints=idv_schedule,
            outside_value=idv_schedule[-1][-1],
            framework=None,
        )
        self._team_entropy_coeff_schedule = PiecewiseSchedule(
            endpoints=team_schedule,
            outside_value=team_schedule[-1][-1],
            framework=None,
        )

    @override(PPOTorchPolicy)
    def on_global_var_update(self, global_vars):
        super(SCPOKLCoeffSchedule, self).on_global_var_update(global_vars)
        if self._idv_entropy_coeff_schedule is not None:
            self.idv_kl_coeff = self._idv_entropy_coeff_schedule.value(
                global_vars["timestep"]
            )
        if self._team_entropy_coeff_schedule is not None:
            self.team_kl_coeff = self._team_entropy_coeff_schedule.value(
                global_vars["timestep"]
            )
        # printPanel({
        #     'global timesteps': global_vars["timestep"],
        #     # 'num_agents': self.config['env_config']['num_agents'],
        #     # 'global timesteps / num_agents': global_vars["timestep"]/self.config['env_config']['num_agents'],
        #     'idv_kl_coeff': self.idv_kl_coeff,
        #     'team_kl_coeff': self.team_kl_coeff
        # }, title='IRATKLCoeffSchedule on_global_var_update()')


class SCPOPolicy(SCPOKLCoeffSchedule, PPOTorchPolicy):
# class SCPOPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        SCPOKLCoeffSchedule.__init__(self, config)
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        # print('model:', self.model)

    @override(PPOTorchPolicy)
    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        return {
            SampleBatch.VF_PREDS: model.value_function(),
            # NEI_VALUES: model.get_nei_value(),
            TEAM_VALUES: model.team_value_function(),
        }

    
    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        '''Args:
            sample_batch: Dict[agent_id, Tuple['default_policy', Policy, SampleBatch]]
        '''
        with torch.no_grad():
            sample_batch[ORIGINAL_REWARDS] = sample_batch[SampleBatch.REWARDS].copy()
            sample_batch[NEI_REWARDS] = sample_batch[SampleBatch.REWARDS].copy() # for init ppo
            # sample_batch[TEAM_REWARDS] = sample_batch[SampleBatch.REWARDS].copy() # for init ppo
            # sample_batch[NEI_VALUES] = self.model.get_nei_value().cpu().detach().numpy().astype(np.float32)

            if episode: # filter _initialize_loss_from_dummy_batch()
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


    def loss(self, model, dist_class, train_batch: SampleBatch):

        # == get logits & dists ==
        idv_logits, state = model(train_batch)
        idv_new_action_dist = dist_class(idv_logits, model)

        team_logits = model.team_policy()
        team_new_action_dist = dist_class(team_logits, model)

        # >──────────────────────────── actor loss ────────────────────────────────<
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

        # == get ratios ==
        idv_new_action_log_probs = idv_new_action_dist.logp(train_batch[SampleBatch.ACTIONS])
        team_new_action_log_probs = team_new_action_dist.logp(train_batch[SampleBatch.ACTIONS])

        idv_p_ratio = torch.exp(idv_new_action_log_probs - train_batch[SampleBatch.ACTION_LOGP])
        idv_team_p_ratio = torch.exp(idv_new_action_log_probs - team_new_action_log_probs.clone().detach())
        team_idv_p_ratio = torch.exp(team_new_action_log_probs - train_batch[SampleBatch.ACTION_LOGP])

        idv_adv = train_batch[Postprocessing.ADVANTAGES]
        team_adv = train_batch[TEAM_ADVANTAGES]


        prev_idv_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_idv_action_dist.kl(idv_new_action_dist)
            mean_kl_idv_policy_old_new = reduce_mean_valid(action_kl)
            warn_if_infinite_kl_divergence(self, mean_kl_idv_policy_old_new)
        else: # default
            mean_kl_idv_policy_old_new = torch.tensor(0.0, device=idv_p_ratio.device)

        idv_new_dist_entropy = idv_new_action_dist.entropy()
        team_new_dist_entropy = team_new_action_dist.entropy()

        # TODO: 提前到 Trainer.traning_step() 中去？
        # == normalize advatages ==
        if self.config['norm_adv']:
            idv_adv = standardized(idv_adv)
            team_adv = standardized(team_adv)

        # == obj ==
        if self.config.get('idv_policy_no_team', False):
            idv_loss = self.get_idv_loss_without_team(idv_p_ratio, idv_team_p_ratio, idv_adv)
        else:
            idv_loss = self.get_idv_loss(idv_p_ratio, idv_team_p_ratio, idv_adv)

        team_loss = self.get_team_loss(team_idv_p_ratio, team_adv)
        
        if torch.isnan(idv_loss).any():
            logger.warning(" ──────────── idv loss has nan ──────────── ")
        if torch.isinf(idv_loss).any():
            logger.warning(" ──────────── idv loss has inf ──────────── ")
        if torch.isnan(team_loss).any():
            logger.warning(" ──────────── team loss has nan ──────────── ")
        if torch.isinf(team_loss).any():
            logger.warning(" ──────────── team loss has inf ──────────── ")

        # == kl ==
        kl_team_idv = dist_class(team_logits.detach(), model).kl(idv_new_action_dist)
        mean_kl_team_idv = reduce_mean_valid(kl_team_idv)
        warn_if_infinite_kl_divergence(self, mean_kl_team_idv)

        kl_idv_team = prev_idv_action_dist.kl(team_new_action_dist)
        mean_kl_idv_team = reduce_mean_valid(kl_idv_team)
        warn_if_infinite_kl_divergence(self, mean_kl_idv_team)
        

        # >──────────────────────────── critic loss ────────────────────────────────<
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

        team_vf_loss = _compute_value_loss(
            current_vf=model.team_value_function(),
            prev_vf=train_batch[TEAM_VALUES],
            value_target=train_batch[TEAM_VALUE_TARGETS]
        )


    # ╭────────────────────────────────── total loss ───────────────────────────────────╮
        total_loss = reduce_mean_valid(
            -idv_loss \
            - team_loss \
            + self.config["vf_loss_coeff"] * idv_vf_loss \
            + self.config["vf_loss_coeff"] * team_vf_loss
        )

        total_loss += self.idv_kl_coeff * mean_kl_team_idv
        total_loss += self.team_kl_coeff * mean_kl_idv_team

        # self.entropy_coeff=0 by default
        total_loss -= self.entropy_coeff * reduce_mean_valid(idv_new_dist_entropy)
        total_loss -= self.entropy_coeff * reduce_mean_valid(team_new_dist_entropy)

        # Add mean_kl_loss (already processed through `reduce_mean_valid`), if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_idv_policy_old_new
    # ╰─────────────────────────────────────────────────────────────────────────────────╯

        # === STATS ===
        model.tower_stats["total_loss"] = total_loss
        # == policy loss ==
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-idv_loss-team_loss)
        model.tower_stats["mean_idv_policy_loss"] = reduce_mean_valid(-idv_loss)
        model.tower_stats["mean_team_policy_loss"] = reduce_mean_valid(-team_loss)

        # == vf loss ==
        model.tower_stats["mean_vf_loss"] = reduce_mean_valid(idv_vf_loss+team_vf_loss)
        model.tower_stats["mean_idv_vf_loss"] = reduce_mean_valid(idv_vf_loss)
        model.tower_stats["mean_team_vf_loss"] = reduce_mean_valid(team_vf_loss)

        # == advantage ==
        model.tower_stats["idv_advantages"] = reduce_mean_valid(idv_adv)
        model.tower_stats["team_advantages"] = reduce_mean_valid(team_adv)

        # == kl divergence ==
        model.tower_stats["kl(team|idv)"] = mean_kl_team_idv
        model.tower_stats["kl(idv|team)"] = mean_kl_idv_team
        model.tower_stats["kl_idv(old|new)"] = mean_kl_idv_policy_old_new

        # == entropy ==
        model.tower_stats["idv_mean_entropy"] = reduce_mean_valid(idv_new_dist_entropy)
        model.tower_stats["team_mean_entropy"] =  reduce_mean_valid(team_new_dist_entropy)

        return total_loss


    @override(PPOTorchPolicy)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        def get_stats(key):
            return torch.mean(torch.stack(self.get_tower_stats(key)))

        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": get_stats("total_loss"),
                # == policy loss ==
                "mean_policy_loss": get_stats("mean_policy_loss"),
                "idv_policy_loss": get_stats("mean_idv_policy_loss"),
                "team_policy_loss": get_stats("mean_team_policy_loss"),
                
                # == vf loss ==
                "mean_vf_loss": get_stats("mean_vf_loss"),
                "idv_vf_loss": get_stats("mean_idv_vf_loss"),
                "team_vf_loss": get_stats("mean_team_vf_loss"),

                # == advantage ==
                "idv_advantages": get_stats("idv_advantages"),
                "team_advantages": get_stats("team_advantages"),

                # == kl divergence ==
                "kl_team_idv": get_stats("kl(team|idv)"),
                "kl_idv_team": get_stats("kl(idv|team)"),
                "kl_idv_old_new": get_stats("kl_idv(old|new)"),
                "idv_kl_coeff": self.idv_kl_coeff,
                "team_kl_coeff": self.team_kl_coeff,


                # == entropy ==
                "idv_mean_entropy": get_stats("idv_mean_entropy"),
                "team_mean_entropy": get_stats("team_mean_entropy"),

                # == for RLlib single ppo ==
                "kl": get_stats("kl_idv(old|new)"),
                "vf_loss": get_stats("mean_idv_vf_loss"),
                "policy_loss": get_stats("mean_idv_policy_loss"),
            }
        )



    def get_idv_loss(self, idv_p_ratio, idv_team_p_ratio, idv_adv):
        # individual actor update
        imp_weights = idv_p_ratio

        # individual actions probs / team actions probs
        so_weights = idv_team_p_ratio

        surr1 = imp_weights * idv_adv
        surr2 = idv_adv * torch.clamp(
            imp_weights, 
            1.0 - self.config["clip_param"], 
            1.0 + self.config["clip_param"]
        ) 
        clp = torch.clamp(
            so_weights, 
            1.0 - self.config['idv_clip_param'], 
            1.0 + self.config['idv_clip_param']
        )

        surr3 = clp * idv_adv
        # if self.idv_clip_use_adv:
        #     tc_flag = team_adv_targ <= 0.
        # else:
        tc_flag = so_weights >= 1.0

        idv_min = torch.min(surr1, surr2)
        # ts31 = surr3 <= idv_min
        # t3_flag = tgs & tls & ((tc_flag & ts31) | (~tc_flag & ~ts31))
        # ts3 = t3_flag.float().sum() / tn
        # tsl3 = surr3.clone().detach()[t3_flag].mean()

        # if self.idv_use_two_clip:
        tc_min = torch.min(idv_min, surr3)
        tc_max = torch.max(idv_min, surr3)
        idv_min = tc_flag.float() * tc_min + (1 - tc_flag.float()) * tc_max

        return idv_min

    def get_idv_loss_without_team(self, idv_p_ratio, idv_team_p_ratio, idv_adv):
        # individual actor update
        imp_weights = idv_p_ratio

        surr1 = idv_adv * imp_weights
        surr2 = idv_adv * torch.clamp(
            imp_weights, 
            1.0 - self.config["clip_param"], 
            1.0 + self.config["clip_param"]
        ) 

        idv_min = torch.min(surr1, surr2)
        return idv_min

    def get_team_loss(self, team_idv_p_ratio, team_adv):
        imp_weights = team_idv_p_ratio
        surr1 = team_adv * imp_weights
        surr2 = team_adv * torch.clamp(
            imp_weights, 
            1.0 - self.config["team_clip_param"], 
            1.0 + self.config["team_clip_param"]
        ) 
        return torch.min(surr1, surr2)


class SCPOTrainer(PPO):
    @classmethod
    def get_default_config(cls):
        return SCPOConfig()

    def get_default_policy_class(self, config):
        assert config["framework"] == "torch"
        return SCPOPolicy

    @ExperimentalAPI
    def training_step(self) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        with self._timers[SAMPLE_TIMER]:
            if self.config.count_steps_by == "agent_steps":
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers,
                    max_agent_steps=self.config.train_batch_size,
                )
            else:
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers, max_env_steps=self.config.train_batch_size
                )

        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["advantages"])
        # Train
        if self.config._enable_learner_api:
            # TODO (Kourosh) Clearly define what train_batch_size
            #  vs. sgd_minibatch_size and num_sgd_iter is in the config.
            # TODO (Kourosh) Do this inside the Learner so that we don't have to do
            #  this back and forth communication between driver and the remote
            #  learner actors.
            is_module_trainable = self.workers.local_worker().is_policy_to_train
            self.learner_group.set_is_module_trainable(is_module_trainable)
            train_results = self.learner_group.update(
                train_batch,
                minibatch_size=self.config.sgd_minibatch_size,
                num_iters=self.config.num_sgd_iter,
            )

        elif self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        if self.config._enable_learner_api:
            # The train results's loss keys are pids to their loss values. But we also
            # return a total_loss key at the same level as the pid keys. So we need to
            # subtract that to get the total set of pids to update.
            # TODO (Kourosh): We should also not be using train_results as a message
            #  passing medium to infer which policies to update. We could use
            #  policies_to_train variable that is given by the user to infer this.
            policies_to_update = set(train_results.keys()) - {ALL_MODULES}
        else:
            policies_to_update = list(train_results.keys())

        # TODO (Kourosh): num_grad_updates per each policy should be accessible via
        # train_results
        
    # ╭─────────────────────────────── Modification ────────────────────────────────╮
        #   Add global env steps:
        global_vars = {
            # "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: self.workers.local_worker().policy_map[pid].num_grad_updates
                for pid in policies_to_update
            },
        }
    # ╰─────────────────────────────────────────────────────────────────────────────╯

        # Update weights - after learning on the local worker - on all remote
        # workers.
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            if self.workers.num_remote_workers() > 0:
                from_worker_or_learner_group = None
                if self.config._enable_learner_api:
                    # sync weights from learner_group to all rollout workers
                    from_worker_or_learner_group = self.learner_group
                self.workers.sync_weights(
                    from_worker_or_learner_group=from_worker_or_learner_group,
                    policies=policies_to_update,
                    global_vars=global_vars,
                )
            elif self.config._enable_learner_api:
                weights = self.learner_group.get_weights()
                self.workers.local_worker().set_weights(weights)

        if self.config._enable_learner_api:

            kl_dict = {}
            if self.config.use_kl_loss:
                for pid in policies_to_update:
                    kl = train_results[pid][LEARNER_RESULTS_KL_KEY]
                    kl_dict[pid] = kl
                    if np.isnan(kl):
                        logger.warning(
                            f"KL divergence for Module {pid} is non-finite, this will "
                            "likely destabilize your model and the training process. "
                            "Action(s) in a specific state have near-zero probability. "
                            "This can happen naturally in deterministic environments "
                            "where the optimal policy has zero mass for a specific "
                            "action. To fix this issue, consider setting `kl_coeff` to "
                            "0.0 or increasing `entropy_coeff` in your config."
                        )

            # triggers a special update method on RLOptimizer to update the KL values.
            additional_results = self.learner_group.additional_update(
                module_ids_to_update=policies_to_update,
                sampled_kl_values=kl_dict,
                timestep=self._counters[NUM_AGENT_STEPS_SAMPLED],
            )
            for pid, res in additional_results.items():
                train_results[pid].update(res)

            return train_results

        # For each policy: Update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                log_once("ppo_warned_lr_ratio")
                and self.config.get("model", {}).get("vf_share_layers")
                and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                log_once("ppo_warned_vf_clip")
                and mean_reward > self.config.vf_clip_param
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results



def compute_advantage(rewards, values, next_value, dones, gamma=0.99, lambda_=0.95, norm=False):
    # 计算TD误差
    deltas = rewards + gamma * next_value * (1 - dones) - values

    # 计算GAE
    advantages = torch.zeros_like(rewards)
    advantages[-1] = deltas[-1]
    for t in reversed(range(len(rewards) - 1)):
        advantages[t] = deltas[t] + gamma * lambda_ * (1 - dones[t]) * advantages[t + 1]

    if norm:

    # 标准化Advantage
        mean = advantages.mean()
        std = advantages.std()
        advantages = (advantages - mean) / (std + 1e-5)

    return advantages


def normalize_advantage(batch: SampleBatch):
    advantage = batch[Postprocessing.ADVANTAGES]
    mean = advantage.mean()
    std = advantage.std()
    batch[Postprocessing.ADVANTAGES] = (advantage - mean) / (std + 1e-5)
    return batch


def get_mean_std_str(tensor) -> str:
    std, mean = torch.std_mean(tensor)
    return f'{mean.item():.3f} / {std.item():.3f}'


__all__ = ['SCPOConfig', 'SCPOTrainer']