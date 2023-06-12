import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.utils.typing import Dict, TensorType, List, Tuple, ModelConfigDict

from modules.attention import MultiHeadSelfAttention
from marlpo.utils.debug import inspect, printPanel

NEIGHBOUR_ACTIONS = "neighbour_actions"

METADRIVE_MULTI_AGENT_OBS_SHAPES = {
    'obs_ego': (9, ),
    'obs_navi': (2, 5),
    'obs_lidar': (72, ),
}

def get_layer(input_dim, output_dim, num_layers=1, layer_norm=False):
    layers = []
    for i in range(num_layers):
        l = nn.Linear(input_dim, output_dim)
        nn.init.orthogonal_(l.weight.data, gain=math.sqrt(2))
        nn.init.zeros_(l.bias.data)
        if i == 0:
            layers += [l, nn.ReLU(inplace=True)]
        else:
            layers += [l, nn.ReLU(inplace=True)]
        if layer_norm:
            layers += [nn.LayerNorm([output_dim])]
    return nn.Sequential(nn.LayerNorm(input_dim), *layers)


class ARAgentwiseObsEncoder(nn.Module):

    def __init__(
            self, 
            obs_dim: int, 
            act_dim: int,
            num_outputs: int, # NOTE: may not equal to act_dim, e.g. 4 != 2
            hidden_dim: int, 
            custom_model_config: dict = {},
        ):
        super().__init__()

        self.obs_dim = obs_dim

        # == 1. embeddings ==
        self.obs_shapes = custom_model_config.get('obs_shapes', METADRIVE_MULTI_AGENT_OBS_SHAPES)
        _obs_dim_check = 0
        for k, shape in self.obs_shapes.items():
            if 'mask' not in k:
                _obs_dim_check += np.product(shape)
                setattr(self, k + '_embedding',
                        get_layer(shape[-1], hidden_dim))
        assert self.obs_dim == _obs_dim_check

        self.act_embedding = get_layer(act_dim, hidden_dim)

        # == 2. MultiHeadSelfAttention ==
        self.attn = MultiHeadSelfAttention(hidden_dim, hidden_dim, 4, entry=0) # TODO
        # self.attn = MultiHeadSelfAttention(hidden_dim, hidden_dim, 4, entry='all')

        # == 3. dense ==
        l = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(l.bias.data)
        nn.init.orthogonal_(l.weight.data, gain=math.sqrt(2))
        self.dense = nn.Sequential(nn.LayerNorm(hidden_dim), 
                                   l,
                                   nn.ReLU(inplace=True),
                                   nn.LayerNorm(hidden_dim))

        # == 4. policy_head ==
        self.policy_head = nn.Linear(hidden_dim, num_outputs) # NOTE output size
        # policy head should have a smaller scale
        nn.init.orthogonal_(self.policy_head.weight.data, gain=0.01)


    # def forward(self, obs, onehot_action, execution_mask=None):
    def forward(
        self,
        # input_dict: Dict[str, TensorType],
        obs: TensorType,
        nei_actions: TensorType,
        execution_mask=None, # TODO
    ) -> Tuple[TensorType, List[TensorType]]:
        ''' args:
                obs: shape(BatchSize, obs_dim)
                nei_actions: Size(BS, num_nei, act_dim), e.g. (1,4,2)
                execution_mask: Size(BS, num_nei), e.g. (1, 4)
        '''
        _msg = {}

        assert obs.shape[-1] == self.obs_dim 
        
        # 1. slicing the obs
        obs_dict = {} # add keys: obs_ego, obs_navi, obs_lidar
        _pre_idx = 0
        for k, shape in self.obs_shapes.items():
            if 'mask' not in k:
                _cur_idx = _pre_idx + np.product(shape)
                obs_dict[k] = obs[:, _pre_idx:_cur_idx].reshape(-1, *shape)
                _pre_idx = _cur_idx
        _msg['obs.shape'] = obs.shape
        # _msg['obs'] = obs
        __msg = {}
        for k, t in obs_dict.items():
            __msg[k] = t.shape
        _msg['obs_dict'] = __msg
        # _msg['obs_dict'] = obs_dict

        '''
        obs_dict: {
          'obs_ego': torch.Size([1, 9]), 
          'obs_navi': torch.Size([1, 2, 5]), 
          'obs_lidar': torch.Size([1, 72])
        } 
        '''

        # 2. get embeddings
        obs_embedding_dict = {}
        for k, x in obs_dict.items():
            if 'mask' not in k:
                assert hasattr(self, k + '_embedding')
                obs_embedding_dict[k] = getattr(self, k + '_embedding')(x)
            else:
                assert k == 'obs_mask'
        obs_embedding_dict['obs_ego'] = obs_embedding_dict['obs_ego'].unsqueeze(-2)
        obs_embedding_dict['obs_lidar'] = obs_embedding_dict['obs_lidar'].unsqueeze(-2)
        x = obs_embedding = torch.cat(list(obs_embedding_dict.values()), -2) # torch.Size([1, 4, 64]): BatchSize x seq_len x embedding_dim

        __msg = {}
        for k, t in obs_embedding_dict.items():
            __msg[k] = t.shape
        _msg['obs_embedding_dict'] = __msg
        _msg['obs_embedding'] = obs_embedding.shape

        # 3. self-attention encoding
        # no other actions
        if execution_mask is None:
            x = self.dense(self.attn(x, mask=None))
        else: 
            # append neighbour actions as a sequence into before obs,
            # 3.1 get action embedding
            act_embedding = self.act_embedding(nei_actions)
            _msg['nei_actions'] = nei_actions
            _msg['act_embedding'] = act_embedding

            # 3.2 concate obs & action embeddings as a squence
            input_seq = torch.cat([x, act_embedding], -2) # (BS, seq_len, embedding_dim)
            _msg['input_seq'] = input_seq

            # 3.3 reshape execution_mask
            execution_mask = torch.cat([
                torch.zeros(x.shape[:-1]), # (BS, obs_seq_len)
                execution_mask
            ], -1) # (BS, obs_seq_len + num_nei_actions) 
            _msg['execution_mask'] = execution_mask
            # print(execution_mask)

            # 3.4 put seq into attention net
            x = self.dense(self.attn(input_seq, mask=execution_mask))

            '''
            act_embedding = self.act_embedding(nei_actions)
            delta = torch.cat([
                torch.zeros(*act_embedding.shape[:-2], 1,
                            act_embedding.shape[-1]).to(act_embedding),
                act_embedding * execution_mask.unsqueeze(-1)
            ], -2)

            if 'obs_mask' in obs.keys():
                x = self.dense(self.attn(x - delta, mask=obs.obs_mask))
            else:
                x = self.dense(self.attn(x - delta, mask=None))
            '''

        _msg['after_dense'] = x
        x = self.policy_head(x)
        _msg['after_policy_head'] = x

        # printPanel(_msg, f'{self.__class__.__name__}.forward()')
        return x


def generate_mask_from_order(agent_order, ego_exclusive):
    """Generate execution mask from agent order.

    Used during autoregressive training.

    Args:
        agent_order (torch.Tensor): Agent order of shape [*, n_agents].

    Returns:
        torch.Tensor: Execution mask of shape [*, n_agents, n_agents - 1].
    """
    shape = agent_order.shape
    n_agents = shape[-1]
    agent_order = agent_order.view(-1, n_agents)
    bs = agent_order.shape[0]

    cur_execution_mask = torch.zeros(bs, n_agents).to(agent_order)
    all_execution_mask = torch.zeros(bs, n_agents, n_agents).to(agent_order)

    batch_indices = torch.arange(bs)
    for i in range(n_agents):
        agent_indices = agent_order[:, i].long()

        cur_execution_mask[batch_indices, agent_indices] = 1
        all_execution_mask[batch_indices, :,
                           agent_indices] = 1 - cur_execution_mask
        all_execution_mask[batch_indices, agent_indices, agent_indices] = 1
    if not ego_exclusive:
        # [*, n_agent, n_agents]
        return all_execution_mask.view(*shape[:-1], n_agents, n_agents)
    else:
        # [*, n_agents, n_agents - 1]
        execution_mask = torch.zeros(bs, n_agents,
                                     n_agents - 1).to(agent_order)
        for i in range(n_agents):
            execution_mask[:, i] = torch.cat([
                all_execution_mask[..., i, :i], all_execution_mask[..., i,
                                                                   i + 1:]
            ], -1)
        return execution_mask.view(*shape[:-1], n_agents, n_agents - 1)
