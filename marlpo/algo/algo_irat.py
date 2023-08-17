from typing import Dict, List, Tuple
import math
import numpy as np
import gymnasium as gym

from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, discount_cumsum, Postprocessing
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.rllib.utils.sgd import standardized
from ray.rllib.utils.torch_utils import explained_variance, sequence_mask, warn_if_infinite_kl_divergence
from ray.rllib.utils.typing import Dict, TensorType, List, Tuple, ModelConfigDict

from .utils import add_neighbour_rewards
from utils.debug import printPanel, reduce_window_width, WINDOWN_WIDTH_REDUCED
reduce_window_width(WINDOWN_WIDTH_REDUCED, __file__)

torch, nn = try_import_torch()


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
NEAREST_NEI_REWARDS = 'nearest_nei_reward'            #  │──> Choose 1 alternatively
ATTENTIVE_ONE_NEI_REWARD = 'attentive_one_nei_reward' #  │
ATTENTIVE_ALL_NEI_REWARD = 'attentive_all_nei_reward' # ─╯


class IRATConfig(PPOConfig):
    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class=algo_class or IRATTrainer)
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
        # self.use_attention = True

        # == SaCo ==
        self.nei_rewards_mode='mean_nei_rewards'
        self.norm_adv=True
        

    def validate(self):
        super().validate()
        assert self["fuse_mode"] in ["mf", "concat", "none"]

        # common
        self.model["custom_model_config"]["num_neighbours"] = self["num_neighbours"]

        # Central Critic
        self.model["custom_model_config"]["use_central_critic"] = self["use_central_critic"]
        self.model["custom_model_config"]["fuse_mode"] = self["fuse_mode"]
        self.model["custom_model_config"]["counterfactual"] = self["counterfactual"]

        # get obs_shape for every env_config for attention encoder
        self.model["custom_model_config"]["env_config"] = self.env_config

        # == IRAT ==
        self.update_from_dict({"model": {"custom_model": "irat_model"}})


class IRATModel(TorchModelV2, nn.Module):
    """Generic fully connected network."""

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
        '''Args:
            hiddens: e.g., [256, 256]
        '''
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

        # res, diff = self._check_params_equal(last_params, layers.state_dict())
        # res = not res
        printPanel(
            {'params updated': res,
             'params slice': list(layers.parameters())[-2][-1][-5:],
             'params grad': grad,
             'grad std/mean': None if grad == None else torch.std_mean(grad),
            }, 
            title=f'check_{name}_params'
        )


# Trick 8: orthogonal initialization
def orthogonal_initializer(gain=1.0):
    def orthogonal_init(weight):
        nn.init.orthogonal_(weight, gain=gain)
    return orthogonal_init

ModelCatalog.register_custom_model("irat_model", IRATModel)


def _compute_advantage(rollout: SampleBatch, rvat, last_r: float, gamma: float = 0.9, lambda_: float = 1.0):
    REWARD, VALUE, ADVANTAGE, TARGET = rvat
    vpred_t = np.concatenate([rollout[VALUE], np.array([last_r])])
    delta_t = (rollout[REWARD] + gamma * vpred_t[1:] - vpred_t[:-1])
    rollout[ADVANTAGE] = discount_cumsum(delta_t, gamma * lambda_)
    rollout[TARGET] = (rollout[ADVANTAGE] + rollout[VALUE]).astype(np.float32)
    rollout[ADVANTAGE] = rollout[ADVANTAGE].astype(np.float32)
    return rollout

# TODO: more precise
class IRATKLCoeffSchedule:
    def __init__(self, config):
        idx_schedule = config['idx_kl_coeff_schedule'] 
        team_schedule = config['team_kl_coeff_schedule'] 
        self.idv_kl_coeff = idx_schedule[0][-1]
        self.team_kl_coeff = team_schedule[0][-1]

        self._idv_entropy_coeff_schedule = PiecewiseSchedule(
            endpoints=idx_schedule,
            outside_value=idx_schedule[-1][-1],
            framework=None,
        )
        self._team_entropy_coeff_schedule = PiecewiseSchedule(
            endpoints=team_schedule,
            outside_value=team_schedule[-1][-1],
            framework=None,
        )

    @override(PPOTorchPolicy)
    def on_global_var_update(self, global_vars):
        super(IRATKLCoeffSchedule, self).on_global_var_update(global_vars)
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
        #     'num_agents': self.config['env_config']['num_agents'],
        #     'global timesteps / num_agents': global_vars["timestep"]/self.config['env_config']['num_agents'],
        #     'idv_kl_coeff': self.idv_kl_coeff,
        #     'team_kl_coeff': self.team_kl_coeff
        # }, title='IRATKLCoeffSchedule on_global_var_update()')


class IRATPolicy(IRATKLCoeffSchedule, PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        IRATKLCoeffSchedule.__init__(self, config)
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


    def loss(self, model, dist_class, train_batch: SampleBatch):
        """
        Compute loss for Proximal Policy Objective.

        PZH: We replace the value function here so that we query the centralized values instead
        of the native value function.
        """
        msg = {}
        msg['train_batch'] = train_batch
        msg['*'] = '*'
        msg['is_single_trajectory'] = train_batch.is_single_trajectory()
        msg['is_training'] = train_batch.is_training
        msg_tr = {}

        # model.check_params_updated('nei_critic')

        # ╭──────────────────────────────────────────────────╮
        # │  ───────────────   actor loss   ───────────────  │
        # ╰──────────────────────────────────────────────────╯
        # == get logits & dists ==
        idv_logits, state = model(train_batch)
        try:
            idv_new_action_dist = dist_class(idv_logits, model)
        except:
            print('====================='*2)
            print('obs', train_batch[SampleBatch.OBS])
            print('action', train_batch[SampleBatch.ACTIONS])
            print('action logp', train_batch[SampleBatch.ACTION_LOGP])
            print('action dist input', train_batch[SampleBatch.ACTION_DIST_INPUTS])
            print('idv_logits', idv_logits)
            print('variables', model.variables())

        team_logits = model.team_policy()
        team_new_action_dist = dist_class(team_logits, model)

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


        # TODO:
        prev_idv_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_idv_action_dist.kl(idv_new_action_dist)
            mean_idv_2_old_policy_kl_loss = reduce_mean_valid(action_kl)
            warn_if_infinite_kl_divergence(self, mean_idv_2_old_policy_kl_loss)
        else:
            mean_idv_2_old_policy_kl_loss = torch.tensor(0.0, device=idv_p_ratio.device)

        idv_new_dist_entropy = idv_new_action_dist.entropy()
        team_new_dist_entropy = team_new_action_dist.entropy()

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
            print(" ──────────── idv loss has nan ──────────── ")
        if torch.isinf(idv_loss).any():
            print(" ──────────── idv loss has inf ──────────── ")

        if torch.isnan(team_loss).any():
            print(" ──────────── team loss has nan ──────────── ")
        if torch.isinf(team_loss).any():
            print(" ──────────── team loss has inf ──────────── ")

        # == kl ==
        kl_team_2_idv = dist_class(team_logits.detach(), model).kl(idv_new_action_dist)
        mean_kl_team_2_idv = reduce_mean_valid(kl_team_2_idv)
        if warn_if_infinite_kl_divergence(self, mean_kl_team_2_idv):
            print('──── warn_if_infinite_kl_divergence ────', 'mean_kl_team_2_idv')

        kl_idv_2_team = prev_idv_action_dist.kl(team_new_action_dist)
        mean_kl_idv_2_team = reduce_mean_valid(kl_idv_2_team)
        if warn_if_infinite_kl_divergence(self, mean_kl_idv_2_team):
            print('──── warn_if_infinite_kl_divergence ────', 'mean_kl_idv_2_team')
        

        # ╭───────────────────────────────────────────────────╮
        # │  ───────────────   critic loss   ───────────────  │
        # ╰───────────────────────────────────────────────────╯
        assert self.config["use_critic"]

        # value_fn_out = model.value_function() # torch.tensor (B, )
        if self.config["huber_value_loss"]:
            # == huber Value Loss ==
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


        # === Total loss ===
        # ╭──────────────────────────────────────────────────╮
        # │  ───────────────   total loss   ───────────────  │
        # ╰──────────────────────────────────────────────────╯
        total_loss = reduce_mean_valid(
            -idv_loss \
            # + self.idv_kl_coeff * kl_team_2_idv \
            - team_loss \
            # + self.team_kl_coeff * kl_idv_2_team \
            + self.config["vf_loss_coeff"] * idv_vf_loss \
            + self.config["vf_loss_coeff"] * team_vf_loss

            # - torch.abs(old_r - nei_r) * svo_loss
            # + torch.pow(advantage, 2)
        )

        total_loss += self.idv_kl_coeff * mean_kl_team_2_idv
        total_loss += self.team_kl_coeff * mean_kl_idv_2_team

        total_loss -= self.entropy_coeff * reduce_mean_valid(idv_new_dist_entropy)
        total_loss -= self.entropy_coeff * reduce_mean_valid(team_new_dist_entropy)

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_idv_2_old_policy_kl_loss


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
        model.tower_stats["kl(team|idv)"] = mean_kl_team_2_idv
        model.tower_stats["kl(idv|team)"] = mean_kl_idv_2_team
        model.tower_stats["kl_idv(old|new)"] = mean_idv_2_old_policy_kl_loss

        # == entropy ==
        model.tower_stats["idv_mean_entropy"] = reduce_mean_valid(idv_new_dist_entropy)
        model.tower_stats["team_mean_entropy"] =  reduce_mean_valid(team_new_dist_entropy)

        # printPanel(msg, "computing value loss")
        # printPanel(msg_tr, "training msg in loss()")

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


    def update_ratio_by_gap(self, ratio, gap, end_ratio):
        tmp = ratio - gap
        if gap >= 0:
            ratio = max(tmp, end_ratio)
        else:
            ratio = min(tmp, end_ratio)
        return ratio

    def update_idv_clip_ratio(self):
        self.idv_clip_ratio = self.update_ratio_by_gap(self.idv_clip_ratio,
                                                       self.idv_clip_gap, self.idv_end_clip_ratio)
        # tmp = self.idv_clip_ratio - self.idv_clip_gap
        # self.idv_clip_ratio = max(self.idv_end_clip_ratio, tmp)

    def update_team_clip_ratio(self):
        self.team_clip_ratio = self.update_ratio_by_gap(self.team_clip_ratio,
                                                        self.team_clip_gap, self.team_end_clip_ratio)
        # tmp = self.team_clip_ratio - self.team_clip_gap
        # self.team_clip_ratio = min(self.team_end_clip_ratio, tmp)

    def update_idv_kl_coef(self):
        self.idv_kl_coef = self.update_ratio_by_gap(self.config.idv_kl_coeff,
                                                    self.idv_kl_anneal_gap, self.idv_kl_end_coeff)
        # tmp = self.idv_kl_coef - self.idv_kl_anneal_gap
        # self.idv_kl_coef = min(self.idv_kl_end_coef, tmp)

    def update_team_kl_coef(self):
        self.team_kl_coef = self.update_ratio_by_gap(self.team_kl_coef,
                                                     self.team_kl_anneal_gap, self.team_kl_end_coef)
        # tmp = self.team_kl_coef - self.team_kl_anneal_gap
        # self.team_kl_coef = max(self.team_kl_end_coef, tmp)


class IRATTrainer(PPO):
    @classmethod
    def get_default_config(cls):
        return IRATConfig()

    def get_default_policy_class(self, config):
        assert config["framework"] == "torch"
        return IRATPolicy



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


__all__ = ['IRATConfig', 'IRATTrainer']