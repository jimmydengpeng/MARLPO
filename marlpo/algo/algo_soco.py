from typing import List, Tuple
import math
import numpy as np
import gym
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import (
    Postprocessing, 
    compute_gae_for_sample_batch, 
    compute_advantages,
    discount_cumsum
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

from models.sc_model import SCModel, FullyConnectedNetwork
from utils.debug import printPanel, reduce_window_width, WINDOWN_WIDTH_REDUCED
reduce_window_width(WINDOWN_WIDTH_REDUCED, __file__)

torch, nn = try_import_torch()

ModelCatalog.register_custom_model("saco_model", SCModel)
ModelCatalog.register_custom_model("fcn_model", FullyConnectedNetwork)

ORIGINAL_REWARDS = "original_rewards"
NEI_REWARDS = "nei_rewards"
SVO = 'svo'
NEXT_VF_PREDS = 'next_vf_preds'
HAS_NEIGHBOURS = 'has_neighbours'
ATTENTION_MAXTRIX = 'attention_maxtrix'

NEI_REWARDS_MODE = 'nei_rewards_mode'

MEAN_NEI_REWARDS = 'mean_nei_rewards'                 # ─╮ 
MAX_NEI_REWARDS = 'max_nei_rewards'                   #  │
NEAREST_NEI_REWARDS = 'nearest_nei_reward'            #  │──> Choose 1 alternatively
ATTENTIVE_ONE_NEI_REWARD = 'attentive_one_nei_reward' #  │
ATTENTIVE_ALL_NEI_REWARD = 'attentive_all_nei_reward' # ─╯



class SOCOConfig(PPOConfig):
    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class=algo_class or SOCOTrainer)
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
        self.use_sa_and_svo=False
        self.use_fixed_svo=False
        self.fixed_svo=math.pi/4
        self.use_social_attention=True # TODO
        self.use_svo=True
        self.svo_mode='full'
        self.nei_rewards_mode='mean_nei_rewards'
        self.sp_select_mode='numerical'
        self.norm_adv=True
        
     

    def validate(self):
        super().validate()
        assert self["fuse_mode"] in ["mf", "concat", "none"]
        assert self[NEI_REWARDS_MODE] in [
            MEAN_NEI_REWARDS,         # ─╮ 
            MAX_NEI_REWARDS,          #  │
            NEAREST_NEI_REWARDS,      #  │──> Choose 1 alternatively
            ATTENTIVE_ONE_NEI_REWARD, #  │
            ATTENTIVE_ALL_NEI_REWARD, # ─╯
        ], self[NEI_REWARDS_MODE] 

        # common
        self.model["custom_model_config"]["num_neighbours"] = self["num_neighbours"]

        # Central Critic
        self.model["custom_model_config"]["use_central_critic"] = self["use_central_critic"]
        self.model["custom_model_config"]["fuse_mode"] = self["fuse_mode"]
        self.model["custom_model_config"]["counterfactual"] = self["counterfactual"]

        # Attention Encoder
        # self.model["custom_model_config"]["use_attention"] = self["use_attention"]

        # get obs_shape for every env_config for attention encoder
        self.model["custom_model_config"]["env_config"] = self.env_config

        self.onehot_attention = self.model["custom_model_config"]["onehot_attention"]

        # from ray.tune.registry import get_trainable_cls
        # env_cls = get_trainable_cls(self.env)
        # printPanel({'env_cls': env_cls}, color='red')
        obs_shape = self.model["custom_model_config"]['env_cls'].get_obs_shape(self.env_config)


        self.model['custom_model_config']['use_social_attention'] = self['use_social_attention']
        
        # Custom Model configs
        if self.use_sa_and_svo:
            self.update_from_dict({"model": {"custom_model": "saco_model"}})
        else:
            self.update_from_dict({"model": {"custom_model": "fcn_model"}})


        msg = {}
        msg['obs_shape'] = obs_shape
        msg['custom_model_config'] = self.model["custom_model_config"]
        # printPanel(msg, title='SAPPOConfig validate', color='green')


class SOCOPolicy(PPOTorchPolicy):
    # @override(PPOTorchPolicy)
    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        if self.config['use_sa_and_svo']:
            return {
                SVO: model.svo_function(),
                SampleBatch.VF_PREDS: model.value_function(),
                ATTENTION_MAXTRIX: model.last_attention_matrix, # (B, H, 1, num_agents)
            }
        else:
            if self.config['use_fixed_svo']:
                return {
                    SampleBatch.VF_PREDS: model.value_function(),
                }
            else:
                return {
                    SampleBatch.VF_PREDS: model.value_function(),
                    SVO: model.svo_function(),
                }


    def add_neighbour_rewards(
        self,
        sample_batch: SampleBatch
    ) -> Tuple[List, List]:
        infos = sample_batch[SampleBatch.INFOS]
        nei_rewards = []
        has_neighbours = []
        # new_rews = []

        for i, info in enumerate(infos):
            assert isinstance(info, dict)
            if NEI_REWARDS not in info:
                # assert sample_batch[SampleBatch.T][i] == i, (sample_batch[SampleBatch.T][i], sample_batch[SampleBatch.AGENT_INDEX])
                nei_rewards.append(0.)
                has_neighbours.append(False)
                # new_rews.append(0)
            else:
                assert NEI_REWARDS in info
                ego_r = sample_batch[SampleBatch.REWARDS][i]
                # == NEI_REWARDS 列表不为空, 即有邻居 ==
                nei_rewards_t = info[NEI_REWARDS][: self.config['num_neighbours']]
                # print('>>>', self.config['num_neighbours'])
                if nei_rewards_t:
                    assert len(info[NEI_REWARDS]) > 0
                    nei_r = 0
                    # 1. == 使用基于规则的邻居奖励 ==
                    if self.config[NEI_REWARDS_MODE] == MEAN_NEI_REWARDS:
                        nei_r = np.mean(nei_rewards_t)
                    elif self.config[NEI_REWARDS_MODE] == MAX_NEI_REWARDS:
                        nei_r = np.max(nei_rewards_t)
                    elif self.config[NEI_REWARDS_MODE] == NEAREST_NEI_REWARDS:
                        nei_r = nei_rewards_t[0]

                    # 2. or == 使用注意力选择一辆车或多辆车 ==
                    else:
                        atn_matrix = sample_batch[ATTENTION_MAXTRIX][i] #每行均为onehot (H, 1, 当时的邻居数量+1)
                        atn_matrix = np.squeeze(atn_matrix) # (H, num_nei+1)

                        # == 使用 one-hot 加权 ==
                        if self.config[NEI_REWARDS_MODE] == ATTENTIVE_ONE_NEI_REWARD:
                            select_mode = self.config['sp_select_mode']
                            assert select_mode in ('numerical', 'bincount')
                            #  == 使用各个头注意的哪个，然后对每个头计数，看哪个被注意次数最多
                            if select_mode == 'bincount':
                                # if self.config['onehot_attention']:
                                index = np.argmax(atn_matrix, axis=-1) # (H, ) # 每行代表每个头不为0的元素所在的index
                                # atn_matrix = np.argmax(atn_matrix, axis=-1)
                                bincount = np.bincount(index) # (H, ) # 代表每行相同index出现的次数
                                frequent_idx = np.argmax(bincount) # 返回次数最多那个index, 范围为 [0, 当时的邻居数量]
                                # 注意每一时刻 ATTENTION_MATRIX 中 oneehot 向量的长度总是比邻居的数量大 1
                                # 如果idx为0则选择的是自己
                            #  == 把每个头的 one-hot 向量的值加起来然后做 argmax，也就是看哪个位置的数值的和最大就选哪个
                            elif select_mode == 'numerical':
                                frequent_idx = np.argmax(np.sum(atn_matrix, axis=0))
                                
                            if frequent_idx == 0:
                                # TODO: add in config!
                                # nei_r = ego_r # 使用自己的奖励
                                nei_r = 0
                            else:
                                # svo = np.squeeze(sample_batch[SVO])[i]
                                # svo = (svo + 1) * np.pi / 4
                                nei_r = nei_rewards_t[frequent_idx-1]
                                # new_r = np.cos(svo) * ego_r + np.sin(svo) * nei_r
                                # new_rews.append(new_r)
                                            
                        # == 使用原始注意力得分加权 == # TODO
                        elif self.config[NEI_REWARDS_MODE] == ATTENTIVE_ONE_NEI_REWARD:
                            total_sums = np.sum(atn_matrix) # (H, num_nei+1) -> ()
                            scores = np.sum(atn_matrix, axis=0) / total_sums # (H, num_nei+1) -> (num_nei+1, )
                            ego_and_nei_r = np.concatenate((np.array([ego_r]), info[NEI_REWARDS]))
                            # nei_r = info[NEI_REWARDS]
                            length = min(len(scores), len(ego_and_nei_r))
                            nei_r = np.sum((scores[:length] * ego_and_nei_r[:length])[1:])

                    nei_rewards.append(nei_r)
                    has_neighbours.append(True)
                
                else: # NEI_REWARDS 列表为空, 即没有邻居
                    assert len(info[NEI_REWARDS]) == 0
                    # 1. == 此时邻居奖励为0 ==
                    nei_rewards.append(0.)
                    # or 2. == 使用自己的奖励当做邻居奖励 ==
                    # or 3. == 使用自己的奖励 ==
                    # new_rews.append(ego_r)
                    has_neighbours.append(False)


        nei_rewards = np.array(nei_rewards).astype(np.float32)
        has_neighbours = np.array(has_neighbours)
        # new_rewards = np.array(new_rews).astype(np.float32)
        # printPanel({'atn_matrix': atn_matrix, 'nei_r': nei_r})

        sample_batch[NEI_REWARDS] = nei_rewards
        sample_batch[HAS_NEIGHBOURS] = has_neighbours

        # printPanel(msg, f'{self.__class__.__name__}.postprocess_trajectory()')
        return nei_rewards, has_neighbours
    

    def clip_svo(self, svo, min_, max_):
        ''' svo in (-oo, +oo)'''
        if isinstance(svo, np.ndarray):
            clip_method = np.clip
        elif isinstance(svo, torch.Tensor):
            clip_method = torch.clamp
        return clip_method(svo * math.pi, min_, max_)

    def prepare_svo(self, svo, min_, max_, mode='full'):
        # for last activation of svo_head is relu, svo in [0, +oo)
        pass

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        ''' args:
                sample_batch:
                    ATTENTION_MAXTRIX: # 每行均为onehot (B, H, 1, num_agents) 
                other_agent_batches: Dict[agent_id, Tuple['default_policy', SAPPOPolicy, SampleBatch]]
        '''
        msg = {}
        msg['sample_batch'] = sample_batch

        with torch.no_grad():
            sample_batch[ORIGINAL_REWARDS] = sample_batch[SampleBatch.REWARDS].copy()
            sample_batch[NEI_REWARDS] = sample_batch[SampleBatch.REWARDS].copy() # for init ppo

            if episode: # filter _initialize_loss_from_dummy_batch()
                msg['*'] = '*'
                msg['agent id'] = f'agent{sample_batch[SampleBatch.AGENT_INDEX][0]}'
                # msg['agent id last'] = f'agent{sample_batch[SampleBatch.AGENT_INDEX][-1]}'

                if self.config['use_svo']:
                    ''' if use svo:
                        1. add neighbour rewards
                        2. use predicted svo to reshape the reward returned by env 
                            to get a new individual reward
                    '''
                    nei_rewards, has_neighbours = self.add_neighbour_rewards(sample_batch)
                    msg = {}
                    msg['has_neighbours %'] = np.sum(has_neighbours) / len(has_neighbours) * 100
                    assert nei_rewards.shape == has_neighbours.shape

                    if self.config['use_sa_and_svo']:
                        svo = np.squeeze(sample_batch[SVO]) # [0, +oo) # (B, )
                        if self.config['svo_mode'] == 'full':
                            svo = self.clip_svo(svo, 0, math.pi/2)
                        elif self.config['svo_mode'] == 'restrict':
                            svo = self.clip_svo(svo, 0, math.pi/4)
                        old_r = sample_batch[SampleBatch.REWARDS]
                        new_r = np.cos(svo) * old_r + np.sin(svo) * nei_rewards

                    # [预设一个社交倾向] 使用固定的svo值， 当有邻居时修改 reward
                    else:
                        if self.config['use_fixed_svo']:
                            svo_fixed = self.config.get('fixed_svo', math.pi/4)
                            ego_rewards = sample_batch[SampleBatch.REWARDS]
                            svo = np.zeros_like(ego_rewards)
                            svo = np.ma.masked_array(svo, mask=has_neighbours).filled(fill_value=svo_fixed)
                            new_r = np.cos(svo) * ego_rewards + np.sin(svo) * nei_rewards
                            msg['new_r - old_r'] = np.sum(new_r - ego_rewards)
                        else:
                            new_r = sample_batch[SampleBatch.REWARDS]

                    # printPanel(msg)

                    sample_batch[ORIGINAL_REWARDS] = sample_batch[SampleBatch.REWARDS].copy()
                    sample_batch[SampleBatch.REWARDS] = new_r

                    # printPanel(msg, f'{self.__class__.__name__}.postprocess_trajectory()')
            ### end if self.config['use_svo']:

            if sample_batch[SampleBatch.DONES][-1]:
                last_r = 0.0
            else:
                last_r = sample_batch[SampleBatch.VF_PREDS][-1]

            # == 记录 rollout 时的 V 值 == 
            vpred_t = np.concatenate([sample_batch[SampleBatch.VF_PREDS], np.array([last_r])])
            sample_batch[NEXT_VF_PREDS] = vpred_t[1:].copy()

            # batch = compute_advantages(
            #     sample_batch,
            #     last_r,
            #     self.config["gamma"],
            #     self.config["lambda"],
            #     use_gae=self.config["use_gae"],
            #     use_critic=self.config.get("use_critic", True)
            # )
            # if self.config['norm_adv']:
            #     normalize_advantage(batch)

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

        # model.check_head_params_updated('policy')
        # model.check_head_params_updated('critic')
        # model.check_head_params_updated('svo')

        # === actor loss ===
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        if self.config['use_sa_and_svo']:
            if not self.config['use_svo']:
                rewards = train_batch[SampleBatch.REWARDS]
            elif self.config['use_svo']:
                # == new svo ==
                svo = model.svo_function() # torch.tensor (B, )

                msg_tr['svo slice'] = svo[-5:]
                msg_tr['svo mean(std)'] = str(torch.std_mean(svo))
                msg_tr["*"] = '*'
                if self.config['svo_mode'] == 'full':
                    svo = self.clip_svo(svo, 0, math.pi/2)
                elif self.config['svo_mode'] == 'restrict':
                    svo = self.clip_svo(svo, 0, math.pi/4)
                # svo = (svo + 1) * torch.pi / 4 # for tanh
                # svo = svo * torch.pi / 2 # for relu
                msg_svo = {}
                msg_svo['svo cos head_5'] = torch.cos(svo)[:5]
                msg_svo['svo sin head_5'] = torch.sin(svo)[:5]
                # msg_svo['*'] = '*'
                # mid = len(svo) / 2
                # msg_svo['svo cos mid'] = torch.cos(svo)[mid:mid+5]
                # msg_svo['svo sin mid'] = torch.sin(svo)[mid:mid+5]
                msg_svo['**'] = '*'
                msg_svo['svo cos last_5'] = torch.cos(svo)[-5:]
                msg_svo['svo sin last_5'] = torch.sin(svo)[-5:]
                # printPanel(msg_svo, 'loss(): svo')

                msg_atn = {}
                msg_atn['atn matrix'] = model.last_attention_matrix[-1][0][0] # (B, H, 1[ego_q], num_agents)
                # printPanel(msg_atn, 'loss(): attention matrix')

                old_r = train_batch[ORIGINAL_REWARDS]
                nei_r = train_batch[NEI_REWARDS] 
                # 重新计算加权奖励
                rewards = torch.cos(svo) * old_r + torch.sin(svo) * nei_r # torch.tensor (B, )
                # msg_tr['new_r mean/std'] = torch.std_mean(new_r)
                # msg_tr['new_r req'] = new_r.requires_grad
                # 如果没有邻居 则用自身的原始奖励
                # TODO: need this?
                # new_r = (1-train_batch[HAS_NEIGHBOURS]) * (1-torch.cos(svo)) * old_r + new_r
        elif not self.config['use_sa_and_svo']:
            if self.config['use_fixed_svo']:
                rewards = train_batch[SampleBatch.REWARDS]
            elif not self.config['use_fixed_svo']:
                svo = model.svo_function() # torch.tensor (B, )
                # model.check_params_updated('_svo')
                # check_svo(svo)
                if self.config['svo_mode'] == 'full':
                    svo = self.clip_svo(svo, 0, math.pi/2)
                elif self.config['svo_mode'] == 'restrict':
                    svo = self.clip_svo(svo, 0, math.pi/4)
                old_r = train_batch[ORIGINAL_REWARDS]
                nei_r = train_batch[NEI_REWARDS] 
                # 重新计算加权奖励
                rewards = torch.cos(svo) * old_r + torch.sin(svo) * nei_r # torch.tensor (B, )
            

        # 计算 GAE Advantage
        values = train_batch[SampleBatch.VF_PREDS]
        next_value = train_batch[NEXT_VF_PREDS]
        dones = train_batch[SampleBatch.TERMINATEDS].to(dtype=values.dtype)
        gamma = self.config['gamma']
        lambda_ = self.config['lambda']

        advantage = compute_advantage(
            rewards, values, next_value, dones, gamma, lambda_, 
            norm=self.config['norm_adv']
        ) # 带svo梯度

        msg_tr['**'] = '*'
        msg_tr['advantage'] = advantage[:5]
        msg_tr['advantage std/mean'] = torch.std_mean(advantage)
        msg_tr['advantage req'] = advantage.requires_grad # True
        # printPanel(msg_tr, raw_output=True)


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


        # 不使用 actor loss 更新 svo
        # if not self.config['use_svo']: 
        #     advantage = train_batch[Postprocessing.ADVANTAGES] 

        surrogate_loss = torch.min(
            advantage * logp_ratio,
            advantage *
            torch.clamp(logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]),
        )

        msg_tr['***'] = '*'
        msg_tr['surrogate_loss mean/std'] = torch.std_mean(surrogate_loss)
        # msg_tr['surrogate_loss slice'] = surrogate_loss[:5] 

        # === Compute a value function loss. ===
        assert self.config["use_critic"]

        value_fn_out = model.value_function() # torch.tensor (B, )
        # model.check_params_updated('svo')
        # model.check_params_updated('policy')
        # model.check_params_updated('value')
       

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

        # == 使用原始 PPO 的 value loss ==
        else:
            # TODO: use ippo v clip
            v_target = advantage.detach() + train_batch[SampleBatch.VF_PREDS]
            train_batch[Postprocessing.VALUE_TARGETS] = v_target
            # value_targets = train_batch[Postprocessing.VALUE_TARGETS]
            # vpred_t = torch.concat((value_fn_out, torch.tensor([last_r])))

            # == 1. use 1-step ted error ==
            # delta_t = q_value - value_fn_out

            # == or 2. use GAE Advantage + V_t == 
            delta_t = (v_target - value_fn_out)

            vf_loss = torch.pow(delta_t, 2.0)
            # vf_loss = torch.pow(value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
            # vf_loss = torch.pow(value_fn_out - value_targets, 2.0)
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])

            msg_tr['vf_loss mean/std'] = torch.std_mean(vf_loss) 
            # msg_tr['vf_loss slice'] = vf_loss[:5] 

        mean_vf_loss = reduce_mean_valid(vf_loss_clipped)

        # === Total loss ===
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


        # printPanel(msg, "computing value loss")
        # printPanel(msg_tr, "training msg in loss()")

        return total_loss


class SOCOTrainer(PPO):
    @classmethod
    def get_default_config(cls):
        return SOCOConfig()

    def get_default_policy_class(self, config):
        assert config["framework"] == "torch"
        return SOCOPolicy



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


def check_svo(svo):
    msg = {}
    msg['svo std/mean'] = torch.std_mean(svo)
    printPanel(msg, title='check svo in loss()')
