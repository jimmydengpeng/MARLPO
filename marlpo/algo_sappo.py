from typing import List
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

from marlpo.models.sa_model import SAModel
from marlpo.utils import rich, inspect, printPanel
rich.get_console().width -= 30

torch, nn = try_import_torch()

ModelCatalog.register_custom_model("sa_model", SAModel)

ORIGINAL_REWARDS = "original_rewards"
NEI_REWARDS = "nei_rewards"
SVO = 'svo'
NEXT_VF_PREDS = 'next_vf_preds'
HAS_NEIGHBOURS = 'has_neighbours'
ATTENTION_MAXTRIX = 'attention_maxtrix'


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
    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        return {
            SVO: model.svo_function(),
            SampleBatch.VF_PREDS: model.value_function(),
            ATTENTION_MAXTRIX: model.last_attention_matrix, # (B, H, 1, num_agents)
        }

    def add_nei_rewards(self, sample_batch, other_agent_batches) -> List:
        # sample_batch["NEI_REWARDS"] =
        for index in range(sample_batch.count):

            environmental_time_step = sample_batch["t"][index]

            # "neighbours" may not be in sample_batch['infos'][index]:
            neighbours = sample_batch['infos'][index].get("neighbours", [])

            nei_r_list = []
            # Note that neighbours returned by the environment are already sorted 
            # based on their distance to the ego vehicle whose info is being used here.
            for nei_count, nei_name in enumerate(neighbours):
                if nei_count >= self.config["num_neighbours"]:
                    break

                nei_r = None
                if nei_name in other_agent_batches:
                    other_agent_batches[nei_name]
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
                    nei_r = nei_batch[SampleBatch.REWARDS][new_index]
                    # nei_act = nei_batch[SampleBatch.ACTIONS][new_index]

                if nei_r is not None:
                    print(sample_batch[SampleBatch.INFOS][index]['agent_id'], '>>> nei:', nei_name, ': nei_r', nei_r)
                    nei_r_list.append(nei_r)
                
            nei_r_sum = np.sum(nei_r_list)
            ego_r = sample_batch[SampleBatch.REWARDS][index]
            # svo = sample_batch["SVO"][index]
            
            # sample_batch[SampleBatch.REWARDS][index] = 

            # sample_batch["NEI_REWARDS"] = [index] = nei_r_list


        # return 



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
            if episode: # filter _initialize_loss_from_dummy_batch()
                msg['*'] = '*'
                msg['agent id'] = f'agent{sample_batch[SampleBatch.AGENT_INDEX][0]}'
                # msg['agent id last'] = f'agent{sample_batch[SampleBatch.AGENT_INDEX][-1]}'

                infos = sample_batch[SampleBatch.INFOS]
                nei_rews = []
                has_neighbours = []
                for i, info in enumerate(infos):
                    assert isinstance(info, dict)
                    if NEI_REWARDS not in info:
                        # assert sample_batch[SampleBatch.T][i] == i, (sample_batch[SampleBatch.T][i], sample_batch[SampleBatch.AGENT_INDEX])
                        nei_rews.append(0.)
                        has_neighbours.append(0)
                        # agent_id = f'agent{sample_batch[SampleBatch.AGENT_INDEX][0]}'
                        # nei_rewards = info[0][agent_id][NEI_REWARDS]
                        # print(nei_rewards)
                    else:
                        assert NEI_REWARDS in info
                        if info[NEI_REWARDS]:
                            # == 1. 使用所有邻居的奖励平均 ==
                            # nei_rews.append(np.mean(info[NEI_REWARDS][: self.config['num_neighbours']])) 

                            # or == 2. 使用注意力选择的一辆车的奖励 ==
                            atn_matrix = sample_batch[ATTENTION_MAXTRIX][i] #每行均为onehot (H, 1, 当时的邻居数量+1)
                            atn_matrix = np.squeeze(atn_matrix) # (H, num_agents)
                            index = np.argmax(atn_matrix, axis=-1) # (H, ) # 每行代表每个头不为0的元素所在的index
                # atn_matrix = np.argmax(atn_matrix, axis=-1)
                            bincount = np.bincount(index) # (H, ) # 代表每行相同index出现的次数
                            frequent_idx = np.argmax(bincount) # 返回次数最多那个index, 范围为 [0, 当时的邻居数量]
                            # 注意每一时刻 ATTENTION_MATRIX 中 oneehot 向量的长度总是比邻居的数量大 1
                            # 如果idx为0则选择的是自己
                            if frequent_idx == 0:
                                # 使用自己的奖励
                                # nei_r = sample_batch[SampleBatch.REWARDS][i]
                                nei_r = 0
                            else:
                                nei_r = info[NEI_REWARDS][frequent_idx-1]

                            # print('>>> ', bincount, frequent_idx, len(info[NEI_REWARDS]), nei_r)

                            nei_rews.append(nei_r)
                            has_neighbours.append(1)
                        else: 
                            nei_rews.append(0.)
                            has_neighbours.append(0)


                nei_rewards = np.array(nei_rews).astype(np.float32)
                # printPanel({'atn_matrix': atn_matrix, 'nei_r': nei_r})

                # nei_r = nei_r * atn_matrix

                sample_batch[NEI_REWARDS] = nei_rewards

                sample_batch[HAS_NEIGHBOURS] = np.array(has_neighbours)

                msg['**'] = '*'

                svo = np.squeeze(sample_batch[SVO]) 

                # === 固定 SVO ===
                # svo = np.zeros_like(svo) + (2/3)

                svo = (svo + 1) * np.pi / 4
                # svo = svo * np.pi / 2


                old_r = sample_batch[SampleBatch.REWARDS]
                new_r = np.cos(svo) * old_r + np.sin(svo) * nei_rewards

                # new_r = (1-sample_batch[HAS_NEIGHBOURS]) * (1-np.cos(svo)) * old_r + new_r

                sample_batch[ORIGINAL_REWARDS] = sample_batch[SampleBatch.REWARDS].copy()
                sample_batch[SampleBatch.REWARDS] = new_r

                # printPanel(msg, f'{self.__class__.__name__}.postprocess_trajectory()')

            if sample_batch[SampleBatch.DONES][-1]:
                last_r = 0.0
            else:
                last_r = sample_batch[SampleBatch.VF_PREDS][-1]
            # print('last_r', last_r)

            # == 记录 rollout 时的 V 值 == 
            vpred_t = np.concatenate([sample_batch[SampleBatch.VF_PREDS], np.array([last_r])])
            sample_batch[NEXT_VF_PREDS] = vpred_t[1:].copy()
           
            batch = compute_advantages(
                sample_batch,
                last_r,
                self.config["gamma"],
                self.config["lambda"],
                use_gae=self.config["use_gae"],
                use_critic=self.config.get("use_critic", True)
            )

        return batch


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
        # model.check_head_params_updated('value')
        # model.check_head_params_updated('svo')
        # model.check_params_updated('')

        # === actor loss ===
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)


        # == new svo ==
        svo = model.svo_function() # torch.tensor (B, )

        msg_tr['svo slice'] = svo[-5:]
        msg_tr['svo mean(std)'] = str(torch.std_mean(svo))
        msg_tr["*"] = '*'


        svo = (svo + 1) * torch.pi / 4 # for tanh
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
        new_r = torch.cos(svo) * old_r + torch.sin(svo) * nei_r # torch.tensor (B, )
        # msg_tr['new_r mean/std'] = torch.std_mean(new_r)
        # msg_tr['new_r req'] = new_r.requires_grad
        # 如果没有邻居 则用自身的原始奖励
        # TODO: need this?
        # new_r = (1-train_batch[HAS_NEIGHBOURS]) * (1-torch.cos(svo)) * old_r + new_r

        msg2 = {}
        msg2['term'] = train_batch[SampleBatch.TERMINATEDS]

        # if train_batch[SampleBatch.TERMINATEDS][-1]:
        #     last_r = 0.0
        # else:
        #     last_r = train_batch[SampleBatch.VF_PREDS][-1]

        # msg['last_r'] = last_r

        # vpred_t = torch.tensor(np.concatenate([train_batch[SampleBatch.VF_PREDS], np.array([last_r])]))
        
        # q_value = new_r + gamma * train_batch[NEXT_VF_PREDS]
        # delta_t = q_value - train_batch[SampleBatch.VF_PREDS]
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        # advantage = discount_cumsum(delta_t, gamma * lambda_)
        # advantage = delta_t

        # 根据新的奖励计算 GAE Advantage
        values = train_batch[SampleBatch.VF_PREDS]
        next_value = train_batch[NEXT_VF_PREDS]
        dones = torch.tensor(train_batch[SampleBatch.TERMINATEDS]).to(dtype=torch.float32)
        gamma = self.config['gamma']
        lambda_ = self.config['lambda']

        advantage = compute_advantage(new_r, values, next_value, dones, gamma, lambda_)
        msg_tr['**'] = '*'
        # msg_tr['advantage'] = advantage[:5]
        # msg_tr['advantage mean/std'] = torch.std_mean(advantage)
        # msg_tr['advantage req'] = advantage.requires_grad # True
        # printPanel(msg2, raw_output=True)


        # msg['has_nei'] = train_batch[HAS_NEIGHBOURS][-5:]
        # msg['old_r'] = old_r[-5:]
        # msg['nei_r_'] = nei_r[-5:]
        # msg['new_r'] = new_r[-5:]
        # msg['*'] = '*'
        # msg['old_r req'] = old_r.requires_grad
        # msg['nei_r req'] = nei_r.requires_grad
        # msg['new_r req'] = new_r.requires_grad
        # msg['**'] = '*'


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
        # advantage = train_batch[Postprocessing.ADVANTAGES] 

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
            # value_targets = train_batch[Postprocessing.VALUE_TARGETS]
            # vpred_t = torch.concat((value_fn_out, torch.tensor([last_r])))
            vpred_t = train_batch[NEXT_VF_PREDS]

            # == 1. use 1-step ted error ==
            # delta_t = q_value - value_fn_out

            # == or 2. use GAE Advantage + V_t == 
            delta_t = (advantage + train_batch[SampleBatch.VF_PREDS] - value_fn_out)

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


        # msg['next_vf_preds'] = train_batch[NEXT_VF_PREDS][-5:]
        # printPanel(msg, "computing value loss")
        # printPanel(msg_tr, "training msg in loss()")

        return total_loss


class SAPPOTrainer(PPO):
    @classmethod
    def get_default_config(cls):
        return SAPPOConfig()

    def get_default_policy_class(self, config):
        assert config["framework"] == "torch"
        return SAPPOPolicy



def compute_advantage(rewards, values, next_value, dones, gamma=0.99, lambda_=0.95):
    # 计算TD误差
    deltas = rewards + gamma * next_value * (1 - dones) - values

    # 计算GAE
    advantages = torch.zeros_like(rewards)
    advantages[-1] = deltas[-1]
    for t in reversed(range(len(rewards) - 1)):
        advantages[t] = deltas[t] + gamma * lambda_ * (1 - dones[t]) * advantages[t + 1]

    # 标准化Advantage
    mean = advantages.mean()
    std = advantages.std()
    advantages = (advantages - mean) / std

    return advantages