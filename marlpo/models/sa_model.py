import logging
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, Tuple, ModelConfigDict

from marlpo.modules import SAEncoder
from modules.models_rlagents import EgoAttentionNetwork
from marlpo.utils import inspect, printPanel

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


CENTRALIZED_CRITIC_OBS = "centralized_critic_obs"
COUNTERFACTUAL = "counterfactual"
NEIGHBOUR_INFOS = "neighbour_infos"
NEIGHBOUR_ACTIONS = "neighbour_actions"
EXECUTION_MASK = 'execution_mask'


# === KEYS ===
EGO_STATE = 'ego_state'
EGO_NAVI = 'ego_navi'
LIDAR = 'lidar'
COMPACT_EGO_STATE = 'compact_ego_state'
COMPACT_NEI_STATE = 'compact_nei_state'
NEI_STATE = 'nei_state'


ATTENTION_HIDDEN_DIM = 64
ATTENTION_HEADS = 4

RELATIVE_OBS_DIM = 91
POLICY_HEAD_ACTIVATION = 'relu'

class SAModel(TorchModelV2, nn.Module):

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # == common attr ==
        self.custom_model_config = model_config['custom_model_config']
        self.obs_dim = int(np.product(self.obs_space.shape))

        # == attention mudule attr ==
        self.use_attention = self.custom_model_config.get("use_attention", False)
        self.hidden_dim = self.custom_model_config.get('attention_dim', ATTENTION_HIDDEN_DIM)
        self.num_neighbours = self.custom_model_config.get('num_neighbours', 0)

        self.use_centralized_critic = self.custom_model_config.get('use_centralized_critic', False)
        self.critic_obs_dim = self.get_critic_obs_dim()

        # == actor ==
        if self.use_attention:
            self.obs_shape = self.custom_model_config['env_cls'].get_obs_shape(self.custom_model_config['env_config'])

            self.attention_backbone = self._get_attention_backbone()
            print('==='*20)
            print(self.attention_backbone)
            print('==='*20)
            self.policy_head = self.get_policy_head(num_outputs, model_config)
            # nn.init.orthogonal_(self.policy_head.weight.data, gain=0.01)
        else:
            self.actor = self._get_fcn_actor(model_config, num_outputs) # TODO 让actor Critic共享中间的特征向量

        # == critic ==
        self._init_critic(model_config)

        # svo net
        self.svo_layer = self.get_policy_head(num_outputs=1, model_config=model_config)

        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_obs_in = None
        # 
        self.last_svo = None

        # TODO === Fix WARNING catalog.py:617 -- Custom ModelV2 should accept all custom options as **kwargs, instead of expecting them in config['custom_model_config']! === 

        if self.use_centralized_critic:
            self.view_requirements[CENTRALIZED_CRITIC_OBS] = ViewRequirement(
                space=Box(obs_space.low[0], obs_space.high[0], shape=(self.critic_obs_dim, ))
            )
        
        self.view_requirements[SampleBatch.ACTIONS] = ViewRequirement(
            space=action_space
        )
        

    def _get_fcn_actor(self, model_config, num_outputs):
        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(model_config.get("post_fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear") # False by default
        self.free_log_std = model_config.get("free_log_std")
        # Generate free-floating bias variables for the second half of the outputs.
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two",
                num_outputs,
            )
            num_outputs = num_outputs // 2

        layers = []
        self._logits = None

        prev_layer_size = int(np.product(self.obs_space.shape)) 

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
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=num_outputs,
                        initializer=normc_initializer(0.01),
                        activation_fn=None
                    )
                )
            else:
                self.num_outputs = ([int(np.product(self.obs_space.shape))] + hiddens[-1:])[-1]

        # Layer to add the log std vars to the state-dependent means.
        if hasattr(self, 'free_log_std') and self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)
        
        return nn.Sequential(*layers)


    def _get_attention_backbone(self) -> nn.Module:
        ''' 该网络会返回一个经过注意力加权后的特征向量，维度为 B x 1 x D_hidden
        '''
        # == get obs_shape for every env_config == 

        # self.custom_model_config['obs_shape'] = self.obs_shape
        input_dim = self.obs_shape[COMPACT_EGO_STATE][0] # 8

        ''' old '''
        # encoder = SAEncoder(self.obs_dim, self.hidden_dim, self.custom_model_config)

        ''' new '''
        atn_net_config = {
            "embedding_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [64, 64],
                "reshape": False,
                "in": input_dim,
            },
            "others_embedding_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [64, 64],
                "reshape": False,
                "in": input_dim,
            },
            "self_attention_layer": None,
            "attention_layer": {
                "type": "EgoAttention",
                "feature_size": 64,
                "heads": ATTENTION_HEADS,
            },
            "output_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [64, 64],
                "reshape": False
        }}

        encoder = EgoAttentionNetwork(atn_net_config)

        return encoder


    def get_policy_head(self, num_outputs, model_config):
        ''' 用于从之前网络中提取到的特征向量(可能再和原始观测输入concat)得到动作输出
        '''
        input_dim = self.hidden_dim + RELATIVE_OBS_DIM
        mlp_hiddens = list(model_config.get("fcnet_hiddens", [])) + list(model_config.get("post_fcnet_hiddens", []))
        hiddens = self.custom_model_config.get('policy_head_hiddens', mlp_hiddens)
        activation = POLICY_HEAD_ACTIVATION

        layers = []
        prev_layer_size = input_dim

        # Create layers 0 to second-last.
        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation
            ))
            prev_layer_size = size
            
        layers.append(
            SlimFC(
                in_size=prev_layer_size,
                out_size=num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None
            ))
        
        return nn.Sequential(*layers)



    def _init_critic(self, model_config):
        '''init _value_branch_separate & _value_branch
            obs -> (_value_branch_separate) --> (_value_branch)
        '''
        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(model_config.get("post_fcnet_hiddens", [])) # [256, 256]
        activation = model_config.get("fcnet_activation")
        self.vf_share_layers = model_config.get("vf_share_layers") # False by default, if True, then use features abstracted from backbone

        prev_vf_layer_size = self.critic_obs_dim

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.

            # === Modification ===
            # NOTE: could be central critic or vanilla critic
            # prev_vf_layer_size = int(np.product(obs_space.shape))
            prev_vf_layer_size = self.critic_obs_dim
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
        
            self.value_head = SlimFC(
                in_size=prev_vf_layer_size, 
                out_size=1, 
                initializer=normc_initializer(0.01), 
                activation_fn=None
            )
        else:
            # prev_vf_layer_size = self.hidden_dim + self.critic_obs_dim
            self.value_head = self.get_policy_head(num_outputs=1, model_config=model_config)


    def get_critic_obs_dim(self):
        if not self.use_centralized_critic:
            if self.use_attention:
                return RELATIVE_OBS_DIM
            else:
                return self.obs_space.shape[0]
        else:
            return get_centralized_critic_obs_dim(
                self.obs_space, self.action_space, self.model_config["custom_model_config"]["counterfactual"],
                self.model_config["custom_model_config"]["num_neighbours"],
                self.model_config["custom_model_config"]["fuse_mode"]
            )

    def get_relative_obs(self, obs_dict):
        entries = [EGO_STATE, EGO_NAVI, LIDAR]
        return torch.concat([obs_dict[k].reshape((obs_dict[k].shape[0], -1)) for k in entries], dim=-1)


    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        '''
        args:
            input_dict.keys():
            [rollout]
                ('obs', 'obs_flat', 'neighbour_actions', 'execution_mask')
            [training]
                ('obs', 'actions', 'rewards', 'terminateds', 'truncateds', 'infos', 'eps_id', 'unroll_id', 'agent_index', 't', 'neighbour_actions', 'action_dist_inputs', 'action_logp', 'centralized_critic_obs', 'vf_preds', 'advantages', 'value_targets', 'obs_flat')

            input_dict['obs_flat'].shape:
            [rollout]
                Size(1, 91)
            [training]
                Size(512:BatchSize, 91)
        '''
        '''
        args:
            [ _initialize_loss_from_dummy_batch() ]
            ╭─────────────────────── input_dict ───────────────────────╮
            │  obs:                                                    │  
            │                torch.Size([32, 131])                     │
            │             or                                           │
            │                compact_ego_state: torch.Size([32, 8])    │
            │                compact_nei_state: torch.Size([32, 4, 8]) │
            │                ego_navi:          torch.Size([32, 2, 5]) │
            │                ego_state:         torch.Size([32, 9])    │
            │                lidar:             torch.Size([32, 72])   │
            │  new_obs:      ndarray.shape=(32, 131)                   │
            │  actions:      ndarray.shape=(32, 2)                     │
            │  prev_actions: ndarray.shape=(32, 2)                     │
            │  rewards:      ndarray.shape=(32,)                       │
            │  prev_rewards: ndarray.shape=(32,)                       │
            │  terminateds:  ndarray.shape=(32,)                       │
            │  truncateds:   ndarray.shape=(32,)                       │
            │  infos:        ndarray.shape=(32,)                       │
            │  eps_id:       ndarray.shape=(32,)                       │
            │  unroll_id:    ndarray.shape=(32,)                       │
            │  agent_index:  ndarray.shape=(32,)                       │
            │  t:            ndarray.shape=(32,)                       │
            │  obs_flat:     torch.Size([32, 131])                     │
            ╰──────────────────────────────────────────────────────────╯

        '''
        msg = {}
        msg['input_dict'] = input_dict
        msg['use_attention'] = self.use_attention
        msg['obs type'] = type(input_dict['obs']).__name__

        

        # attention actor
        if self.use_attention:
            obs = input_dict["obs"] # Dict, str -> (BatchSize, *)
            assert isinstance(obs, dict)
            self._last_obs_in = o = self.get_relative_obs(obs)

            ego_x = obs[COMPACT_EGO_STATE].unsqueeze(-2) # B x 1 x D_absolute_state_dim
            nei_x = obs[COMPACT_NEI_STATE] # B x num_nei x D_absolute_state_dim
            x = torch.concat((ego_x, nei_x), dim=-2)


            x = self.attention_backbone(x) # B x H_hidden
            msg['after attention'] = x
            # x = x.squeeze(dim=-2)
            # msg['after squeeze'] = x
            self._features = x
            
            # o = torch.concat((
            #     obs[EGO_STATE], 
            #     obs[EGO_NAVI].reshape(obs[EGO_NAVI].shape[0], -1), 
            #     obs[LIDAR]), -1)
            x = torch.concat((x, o), -1) # B x (D_hidden + D_obs)

            msg['after concat'] = x
            logits = self.policy_head(x)
            msg['after policy_head'] = logits

            self.last_svo = self.svo_layer(x)

        # mlp actor
        else: 
            obs_flat = input_dict["obs_flat"] # (BatchSize, OBS_DIM)
            self._last_obs_in = obs_flat # (BatchSize, OBS_DIM)
            logits = self.actor(obs_flat)
            # self._features = logits


        if hasattr(self, 'free_log_std') and self.free_log_std: # False by default
            logits = self._append_free_log_std(logits)

        # printPanel(msg, f'{self.__class__.__name__}.forward()')
        return logits, state


    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        if not self.use_centralized_critic:
            if self._value_branch_separate:
                out = self.value_head(
                    self._value_branch_separate(self._last_obs_in)
                ).squeeze(1)
            else:
                assert self._features is not None, "must call forward() first"
                x = torch.concat((self._last_obs_in, self._features), -1) # B x (D_hidden + D_obs)
                out = self.value_head(x).squeeze(1)
            return out
        else:
            raise NotImplementedError


    # @override(TorchModelV2)
    # def value_function(self) -> TensorType:
    #     raise ValueError(
    #         "Centralized Value Function should not be called directly! "
    #         "Call central_value_function(cobs) instead!"
    #     )

    # TODO
    def central_value_function(self, obs):
        assert self.value_head is not None
        return torch.reshape(self.value_head(self._value_branch_separate(obs)), [-1])




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


