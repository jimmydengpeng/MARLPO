import math
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.utils.typing import Dict, TensorType, List, Tuple, ModelConfigDict

from .attention import MultiHeadSelfAttention
from modules.models_rlagents import MultiLayerPerceptron

from utils.debug import inspect, printPanel


# NEIGHBOUR_ACTIONS = "neighbour_actions"

# METADRIVE_MULTI_AGENT_OBS_SHAPES = {
#     'obs_ego': (9, ),
#     'obs_navi': (2, 5),
#     'obs_lidar': (72, ),
# }

DEFAULT_OBS_SHAPE = {
    'ego_state': (9, ),
    'ego_navi': (2, 5),
    'lidar': (72, ),
}

# EMBEDDINGS = {
#     'state': 21,
# } # how many embedding laysers should init

OBS_TO_EMBEDDINGS = {
    'ego_state': 'state_embedding',
    'nei_state': 'state_embedding',
}

# === KEYS ===
EGO_STATE = 'ego_state'
EGO_NAVI = 'ego_navi'
LIDAR = 'lidar'
COMPACT_EGO_STATE = 'compact_ego_state'
COMPACT_NEI_STATE = 'compact_nei_state'
NEI_STATE = 'nei_state'



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


class SAEncoder(nn.Module):

    def __init__(
            self, 
            obs_dim: int, 
            hidden_dim: int, 
            custom_model_config: dict = {},
        ):
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.obs_shape = custom_model_config.get('obs_shape', DEFAULT_OBS_SHAPE)
        self.num_neighbours = custom_model_config['num_neighbours']
        self.validate()

        # printPanel(custom_model_config, title='SAEncoder init')

        # == 1. Set embeddings ==
        self.set_emdedding_layers()

        # == 2. MultiHeadSelfAttention ==
        # self.attn = MultiHeadSelfAttention(hidden_dim, hidden_dim, 4, entry=self.compute_obs_seq_len()) # TODO
        self.attn = MultiHeadSelfAttention(hidden_dim, hidden_dim, n_heads=4, entry=0) # TODO
        # self.attn = MultiHeadSelfAttention(hidden_dim, hidden_dim, 4, entry='all')


        # == 3. dense ==
        l = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(l.bias.data)
        nn.init.orthogonal_(l.weight.data, gain=math.sqrt(2))
        self.dense = nn.Sequential(nn.LayerNorm(hidden_dim), 
                                   l,
                                   nn.ReLU(inplace=True),
                                   nn.LayerNorm(hidden_dim))

        # === 4. layer_norm ===
        self.layer_norm = nn.LayerNorm(hidden_dim)


    def validate(self):
        # check dims
        assert  np.sum([np.product(shape) for shape in self.obs_shape.values()]) == self.obs_dim
        if NEI_STATE in self.obs_shape:
            assert self.obs_shape[NEI_STATE][0] == self.num_neighbours
        if COMPACT_NEI_STATE in self.obs_shape:
            assert self.obs_shape[COMPACT_NEI_STATE][0] == self.num_neighbours


    def set_emdedding_layers(self):
        input_dim = self.obs_shape[COMPACT_EGO_STATE][0] # 8
        self.ego_embedding = get_layer(input_dim, self.hidden_dim)
        self.nei_embedding = get_layer(input_dim, self.hidden_dim)

        # embedding_config = {
        #     'layer': input_dim,
        # }
        # MultiLayerPerceptron()

       

    def compute_obs_seq_len(self):
        res = 0
        for k, shape in self.obs_shape.items():
            res += len(shape)
        return res
            

    def forward(
        self,
        obs: Dict[str, TensorType], 
        execution_mask=None, # TODO
    ) -> TensorType:
        ''' args:
                obs: shape(BatchSize, obs_dim)
                execution_mask: Size(BS, num_nei), e.g. (1, 4)
        '''
        _msg = {}
        _msg['obs'] = obs

    
        # 1. split obs & get embeddings
        ego_x = self.ego_embedding(obs[COMPACT_EGO_STATE].unsqueeze(-2)) # B x 1 x D_hidden
        nei_x = self.nei_embedding(obs[COMPACT_NEI_STATE]) # B x num_nei x D_hidden
        x = torch.concat((ego_x, nei_x), dim=-2)

        _msg['ego_x_embedding'] = ego_x
        _msg['nei_x_embedding'] = nei_x
        _msg['x embedding'] = x

        # 2. self-attention encoding
        if execution_mask is None:
            x = self.attn(x, mask=None) # scores (B x D_hidden)
            _msg['after multi-head attention'] = x

            x = self.dense(x)
            _msg['after_dense'] = x

        else: 
            raise NotImplementedError

        x = self.layer_norm(x + ego_x.squeeze(-2))

        _msg['after add_and_layer_norm'] = x
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
