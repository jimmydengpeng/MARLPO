import functools

import gymnasium as gym
import numpy as np

from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.core.rl_module import RLModule
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.utils.annotations import (
    DeveloperAPI,
    OverrideToImplementCustomLogic,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
    is_overridden,
    override,
)
from ray.rllib.utils.typing import ModelConfigDict

from marlpo.model_arippo import ARFullyConnectedModel

from utils.utils import log, inspect

torch, nn = try_import_torch()


class ARIPPOConfig(PPOConfig):
    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class=algo_class or ARIPPOTrainer)

        # self.sgd_minibatch_size = 512

        # self.rollout_fragment_length = 200
        # self.train_batch_size = 2000

        # self.num_sgd_iter = 5
        # self.lr = 3e-4
        # self.clip_param = 0.2
        # self.lambda_ = 0.95

        # self.num_cpus_per_worker = 0.2
        # self.num_cpus_for_local_worker = 1

        # # New RLLib keys
        # self.num_rollout_workers = 5
        # # self.framework = "torch"
        # self.framework_str = "torch"

        # Two important updates
        self.vf_clip_param = 100
        self.old_value_loss = True
        self.update_from_dict({"model": {"custom_model": "ar_model"}})
    '''
    def validate(self):
        # Note that in new RLLib the rollout_fragment_length will auto adjust to a new value
        # so that one pass of all workers and envs will collect enough data for a train batch.
        super().validate()

        from ray.tune.registry import _global_registry, ENV_CREATOR
        from metadrive.constants import DEFAULT_AGENT

        env_class = _global_registry.get(ENV_CREATOR, self["env"])
        single_env = env_class(self["env_config"])

        if "agent0" in single_env.observation_space.spaces:
            obs_space = single_env.observation_space["agent0"]
            act_space = single_env.action_space["agent0"]
        else:
            obs_space = single_env.observation_space[DEFAULT_AGENT]
            act_space = single_env.action_space[DEFAULT_AGENT]

        assert isinstance(obs_space, gym.spaces.Box)
        # assert isinstance(act_space, gym.spaces.Box)
        # Note that we can't set policy name to "default_policy" since by doing so
        # ray will record wrong per agent episode reward!
        self.update_from_dict(
            {
                "multiagent": dict(
                    # Note that we have to use "default" because stupid RLLib has bug when
                    # we are using "default_policy" as the Policy ID.
                    policies={"default": PolicySpec(None, obs_space, act_space, {})},
                    policy_mapping_fn=lambda x: "default"
                )
            }
        )
    '''




ModelCatalog.register_custom_model("ar_model", ARFullyConnectedModel)




class ARIPPOPolicy(PPOTorchPolicy):
    


    @with_lock
    def _compute_action_helper(
        self, input_dict, state_batches, seq_lens, explore, timestep
    ):
        """Shared forward pass logic (w/ and w/o trajectory view API).

        Returns:
            A tuple consisting of 
                a) actions, -> torch.Size([n_agents, act_dim])
                b) state_out, -> []
                c) extra_fetches. -> {'vf_preds': tensor, 'action_dist_inputs': ...}
            The input_dict is modified in-place to include a numpy copy of the computed
            actions under `SampleBatch.ACTIONS`.
        """
        # print('>>> [ARPPOPolicy] _compute_action_helper()...')

        explore = explore if explore is not None else self.config["explore"] # True
        timestep = timestep if timestep is not None else self.global_timestep
        self._is_recurrent = state_batches is not None and state_batches != []

        # Switch to eval mode.
        if self.model:
            self.model.eval()
        
        extra_fetches = {}
        if isinstance(self.model, RLModule):
            if explore:
                fwd_out = self.model.forward_exploration(input_dict)
            else:
                fwd_out = self.model.forward_inference(input_dict)
            # anything but action_dist and state_out is an extra fetch
            action_dist = fwd_out.pop("action_dist")

            if explore:
                actions, logp = action_dist.sample(return_logp=True)
            else:
                actions = action_dist.sample()
                logp = None
            state_out = fwd_out.pop("state_out", {})
            extra_fetches = fwd_out
            dist_inputs = None
        elif is_overridden(self.action_sampler_fn):
            action_dist = dist_inputs = None
            actions, logp, state_out = self.action_sampler_fn(
                self.model,
                obs_batch=input_dict,
                state_batches=state_batches,
                explore=explore,
                timestep=timestep,
            )
        else:
            use_ar_policy = False
            # Call the exploration before_compute_actions hook.
            self.exploration.before_compute_actions(explore=explore, timestep=timestep)
            if is_overridden(self.action_distribution_fn):
                dist_inputs, dist_class, state_out = self.action_distribution_fn(
                    self.model,
                    obs_batch=input_dict,
                    state_batches=state_batches,
                    seq_lens=seq_lens,
                    explore=explore,
                    timestep=timestep,
                    is_training=False,
                )
            else:
                # print("start to compute actions auoregressively from the model <{}>!".format(self.model.name))
                # log(self.model)
                # === Implement the Auto-Regressive Policy here! ===
                use_ar_policy = True
                dist_class = self.dist_class
                
                # dist_inputs, state_out = self.model(input_dict, state_batches, seq_lens)

                # input_dict.keys(): ['obs', 'new_obs', 'actions', 'prev_actions', 'rewards', 'prev_rewards', 'terminateds', 'truncateds', 'infos', 'eps_id', 'unroll_id', 'agent_index', 't']) -> firts 32 test?
                # sample时只有'obs'可用

                # print(input_dict.keys())

                single_agent_input_dict = input_dict.copy()
                obs = single_agent_input_dict[SampleBatch.OBS] # size(n_agents, obs_dim)
                obs_in_chunk = torch.split(obs, 1)


                # log('model', self.config['model'])
                n_other_actions = self.config['model']['custom_model_config']['n_actions']
                actions_input = torch.from_numpy(np.zeros((n_other_actions, np.product(self.action_space.shape))))
                dist_inputs = []
                actions = []
                logp = []
                vf_preds = []
                # assert len(obs_in_chunk) <= n_other_actions+1, (len(obs_in_chunk), n_other_actions)
                for agent_id, o in enumerate(obs_in_chunk):
                    # o = torch.unsqueeze(o, 0)
                    # print(o.shape)
                    single_agent_input_dict[SampleBatch.OBS] = o
                    single_agent_input_dict['other_actions'] = actions_input
                    # pass other agents' actions to model

                    assert state_batches == []
                    assert seq_lens == None
                    dist_input, state_out = self.model(single_agent_input_dict, state_batches, seq_lens)
                    dist_inputs.append(dist_input)
                    assert state_out == []

                    # compute vf
                    vf_preds.append(self.model.value_function())

                    _action_dist = dist_class(dist_input, self.model)
                    # action_dist.append(_action_dist)

                    # Get the exploration action from the forward results.
                    action, _logp = self.exploration.get_exploration_action(
                        action_distribution=_action_dist, timestep=timestep, explore=explore
                    )
                    # print("agent_id:", agent_id, "action", action)
                    actions.append(action)
                    assert isinstance(action, torch.Tensor)
                    if agent_id < len(actions_input): # TODO
                        actions_input[agent_id] = action
                    logp.append(_logp)
                
                dist_inputs = torch.squeeze(torch.stack(dist_inputs), 1)
                action_dist = dist_class(dist_inputs, self.model)
                actions = torch.squeeze(torch.stack(actions), 1)
                logp = torch.squeeze(torch.stack(logp), 1)
                # actions = np.array(actions)
                # logp = np.array(logp_)
                extra_fetches[SampleBatch.VF_PREDS] = torch.squeeze(torch.stack(vf_preds), 1)
                # print('extra_fetches ==>', extra_fetches)
                # print('[ARIPPOPolicy] sampled actions:', actions)

            if not (
                isinstance(dist_class, functools.partial)
                or issubclass(dist_class, TorchDistributionWrapper)
            ):
                raise ValueError(
                    "`dist_class` ({}) not a TorchDistributionWrapper "
                    "subclass! Make sure your `action_distribution_fn` or "
                    "`make_model_and_action_dist` return a correct "
                    "distribution class.".format(dist_class.__name__)
                )
            
            # 如果不使用AR-Policy
            if not use_ar_policy:
                print(f"[Warning] {__name__}: Not using AR-Policy!!!")
                action_dist = dist_class(dist_inputs, self.model)

                # Get the exploration action from the forward results.
                actions, logp = self.exploration.get_exploration_action(
                    action_distribution=action_dist, timestep=timestep, explore=explore
                ) # size(n_agents, act_dim), size(n_agents)

        # Add default and custom fetches.
        # 计算Value_function
        if not extra_fetches:
            assert not use_ar_policy
            # input_dict: SampleBatch(n_agents: ['obs'])
            # state_batches: []
            # action_dist: TorchDiagGaussian
            extra_fetches = self.extra_action_out(
                input_dict, state_batches, self.model, action_dist # action_dist 未使用？
            )

        # Action-dist inputs.
        if dist_inputs is not None:
            extra_fetches[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs

        # Action-logp and action-prob.
        if logp is not None:
            extra_fetches[SampleBatch.ACTION_PROB] = torch.exp(logp.float())
            extra_fetches[SampleBatch.ACTION_LOGP] = logp

        # Update our global timestep by the batch size.
        self.global_timestep += len(input_dict[SampleBatch.CUR_OBS])
        # print('actions:', actions)
        return convert_to_numpy((actions, state_out, extra_fetches))


    @override(PPOTorchPolicy)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        # print('>>> in postprocess_trajectory')
        # inspect(sample_batch)
        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)

class ARIPPOTrainer(PPO):
    @classmethod
    def get_default_config(cls):
        return ARIPPOConfig()

    def get_default_policy_class(self, config):
        assert config["framework"] == "torch"
        return ARIPPOPolicy

