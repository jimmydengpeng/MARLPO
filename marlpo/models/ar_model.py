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

from marlpo.modules import ARAgentwiseObsEncoder
from marlpo.utils.debug import inspect, printPanel

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


CENTRALIZED_CRITIC_OBS = "centralized_critic_obs"
COUNTERFACTUAL = "counterfactual"
NEIGHBOUR_INFOS = "neighbour_infos"
NEIGHBOUR_ACTIONS = "neighbour_actions"
EXECUTION_MASK = 'execution_mask'



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


class ARModel(TorchModelV2, nn.Module):

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        '''
        
        '''
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.custom_model_config = model_config['custom_model_config']
        # == only for ar model ==
        self.use_attention = self.custom_model_config.get("use_attention", False)
        self.num_neighbours = model_config['custom_model_config'].get('num_neighbours', 0)
        self.centralized_critic_obs_dim = self.get_centralized_critic_obs_dim()

        if self.use_attention:
            self.actor = self.get_attention_actor(model_config, num_outputs)
        else:
            self.actor = self.get_fcn_actor(model_config, num_outputs)

        self.critic = self.get_critic(model_config)

        # TODO === Fix WARNING catalog.py:617 -- Custom ModelV2 should accept all custom options as **kwargs, instead of expecting them in config['custom_model_config']! === 


        self.view_requirements[CENTRALIZED_CRITIC_OBS] = ViewRequirement(
            space=Box(obs_space.low[0], obs_space.high[0], shape=(self.centralized_critic_obs_dim, ))
        )
        self.view_requirements[SampleBatch.ACTIONS] = ViewRequirement(
            space=action_space
        )
        # self.view_requirements[NEIGHBOUR_INFOS] = ViewRequirement(
        #     # data_col=SampleBatch.ACTIONS, 
        #     space=Box(action_space.low[0], action_space.high[0], shape=(self.num_neighbours*action_space.shape[0], )),
        #     # used_for_compute_actions=False
        # )
        # self.view_requirements[SampleBatch.INFOS].used_for_compute_actions = True
        # self.view_requirements[NEIGHBOUR_INFOS] = ViewRequirement()
        self.view_requirements[NEIGHBOUR_ACTIONS] = ViewRequirement(
            space=Box(action_space.low[0], action_space.high[0], shape=(self.num_neighbours*action_space.shape[0], )), 
            used_for_compute_actions=False
        )
        self.view_requirements[EXECUTION_MASK] = ViewRequirement(
            space=Box(action_space.low[0], action_space.high[0], shape=(self.num_neighbours, )), 
            used_for_compute_actions=False
        )

    def get_fcn_actor(self, model_config, num_outputs):
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

        # obs.size加入动作size
        prev_layer_size = int(np.product(self.obs_space.shape)) + self.num_neighbours*self.action_space.shape[0]

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
        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)
        
        return nn.Sequential(*layers)


    def get_attention_actor(self, model_config, num_outputs):
        obs_dim = int(np.product(self.obs_space.shape))
        act_dim = self.action_space.shape[0]
        hidden_dim = self.custom_model_config.get('attention_dim', 64)
        encoder = ARAgentwiseObsEncoder(obs_dim, act_dim, num_outputs, hidden_dim, self.custom_model_config)
        return encoder


    def get_critic(self, model_config):
        # NOTE: could be central critic or vanilla critic
        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(model_config.get("post_fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        self.vf_share_layers = model_config.get("vf_share_layers") # False b default

        # === CC Modification: We compute the centralized critic obs size here! ===
        custom_model_config = model_config.get('custom_model_config', {})
        self.fuse_mode = custom_model_config.get("fuse_mode", None)
        self.counterfactual = custom_model_config.get("counterfactual", False)
        # === CCModel get custom_model_config Ends ===

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.

            # === Our Modification ===
            # NOTE: We use centralized critic obs size as the input size of critic!
            # prev_vf_layer_size = int(np.product(obs_space.shape))
            prev_vf_layer_size = self.centralized_critic_obs_dim
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

        self._value_branch = SlimFC(
            in_size=prev_vf_layer_size, 
            out_size=1, 
            initializer=normc_initializer(0.01), 
            activation_fn=None
        )


    def get_centralized_critic_obs_dim(self):
        return get_centralized_critic_obs_dim(
            self.obs_space, self.action_space, self.model_config["custom_model_config"]["counterfactual"],
            self.model_config["custom_model_config"]["num_neighbours"],
            self.model_config["custom_model_config"]["fuse_mode"]
        )


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
                ('obs', 'neighbour_actions', 'obs_flat')
            [training]
                ('obs', 'actions', 'rewards', 'terminateds', 'truncateds', 'infos', 'eps_id', 'unroll_id', 'agent_index', 't', 'neighbour_actions', 'action_dist_inputs', 'action_logp', 'centralized_critic_obs', 'vf_preds', 'advantages', 'value_targets', 'obs_flat')

            input_dict['obs_flat'].shape:
            [rollout]
                Size(1, 91)
            [training]
                Size(512:BatchSize, 91)
        '''
        msg = {}
        msg['input_dict'] = input_dict
        # printPanel(msg, f'{self.__class__.__name__}.forward()')

        assert NEIGHBOUR_ACTIONS in input_dict
        nei_actions = input_dict[NEIGHBOUR_ACTIONS] # shape(BatchSize, num_nei, act_dim)
        execution_mask = input_dict[EXECUTION_MASK]


        # attention actor
        if self.use_attention:
            # nei_actions = nei_actions.view(-1, self.num_neighbours, self.action_space.shape[0]) # (BS, num_nei, act_dim)
            logits = self.actor(input_dict, nei_actions, execution_mask)

        # mlp actor
        else: 
            obs = input_dict["obs_flat"].float() # shape(BatchSize, 91)
            # obs = obs.reshape(obs.shape[0], -1)
            obs = obs.view(obs.shape[0], -1) # size(BatchSize, *)

            # flat actions
            nei_actions_flat = torch.flatten(nei_actions, start_dim=-2) # shape(1, num_nei x 2)

            obs_cat = torch.cat((obs, nei_actions_flat), -1) # shape (1, 91 + action_dim*num_neighbours) == (1, 99)
            logits = self.actor(obs_cat)
            if self.free_log_std: # False by default
                logits = self._append_free_log_std(logits)

        return logits, state



    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        raise ValueError(
            "Centralized Value Function should not be called directly! "
            "Call central_value_function(cobs) instead!"
        )

    def central_value_function(self, obs):
        assert self._value_branch is not None
        return torch.reshape(self._value_branch(self._value_branch_separate(obs)), [-1])
