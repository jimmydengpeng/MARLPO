import gym
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import (
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)

torch, nn = try_import_torch()


class IPPOConfig(PPOConfig):
    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class=algo_class or IPPOTrainer)

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


class IPPOPolicy(PPOTorchPolicy):
    def loss(self, model, dist_class, train_batch):
        """
        Compute loss for Proximal Policy Objective.

        PZH: We replace the value function here so that we query the centralized values instead
        of the native value function.
        """

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

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES] *
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


class IPPOTrainer(PPO):
    @classmethod
    def get_default_config(cls):
        return IPPOConfig()

    def get_default_policy_class(self, config):
        assert config["framework"] == "torch"
        return IPPOPolicy


# ========== Test scripts ==========
def _test():
    # Testing only!
    from marlpo.callbacks import MultiAgentDrivingCallbacks
    from marlpo.train import train
    from copo.torch_copo.utils.utils import get_train_parser
    from metadrive.envs.marl_envs import MultiAgentIntersectionEnv
    from copo.torch_copo.utils.env_wrappers import get_lcf_env, get_rllib_compatible_env

    parser = get_train_parser()
    args = parser.parse_args()
    stop = {"timesteps_total": 200_0000}
    exp_name = "test_mappo" if not args.exp_name else args.exp_name
    config = dict(
        env=get_rllib_compatible_env(get_lcf_env(MultiAgentIntersectionEnv)),
        env_config=dict(
            # start_seed=tune.grid_search([5000]),
            # num_agents=8,
        ),
        # num_sgd_iter=1,
        # rollout_fragment_length=200,
        # train_batch_size=400,
        # sgd_minibatch_size=200,
        num_workers=0,
        # **{USE_CENTRALIZED_CRITIC: True},
        # fuse_mode=tune.grid_search(["concat", "mf"])
        # fuse_mode=tune.grid_search(["none"])
        train_batch_size=100,
        rollout_fragment_length=20,
        sgd_minibatch_size=30,
        old_value_loss=True
    )
    results = train(
        IPPOTrainer,
        config=config,  # Do not use validate_config_add_multiagent here!
        checkpoint_freq=0,  # Don't save checkpoint is set to 0.
        keep_checkpoints_num=0,
        stop=stop,
        num_gpus=args.num_gpus,
        num_seeds=1,
        max_failures=0,
        exp_name=exp_name,
        custom_callback=MultiAgentDrivingCallbacks,
        test_mode=True,
        local_mode=True
    )

def _my_test():
    from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv
    from marlpo.callbacks import MultiAgentDrivingCallbacks
    from marlpo.train.train import train
    from marlpo.env.env_wrappers import get_rllib_compatible_gymnasium_api_env

    stop = {"training_iteration": 1}
    exp_name = "TEST"
    env = get_rllib_compatible_gymnasium_api_env(MultiAgentRoundaboutEnv)
    env_config = dict(
        use_render=False,
        # num_agents=40,
        return_single_space=True,
        # "manual_control": True,
        # crash_done=True,
        # "agent_policy": ManualControllableIDMPolicy
        start_seed=5000,
    )
    ppo_config = (
        PPOConfig()
        .framework('torch')
        .resources(num_gpus=0)
        .rollouts(
            num_rollout_workers=1,
            # rollout_fragment_length=200 # can get from algo?
        )
        .callbacks(MultiAgentDrivingCallbacks)
        .training(
            train_batch_size=1024,
            gamma=0.99,
            lr=3e-4,
            sgd_minibatch_size=512,
            num_sgd_iter=5,
            lambda_=0.95,
        )
        .multi_agent(
        )
        # .evaluation(
        #     evaluation_interval=2,
        #     evaluation_duration=40,
        #     evaluation_config=dict(env_config=dict(environment_num=200, start_seed=0)),
        #     evaluation_num_workers=1,)
        .environment(env=env, render_env=False, env_config=env_config, disable_env_checking=False)
    )


    train(
        IPPOTrainer,
        config=ppo_config,
        stop=stop,
        exp_name=exp_name,
        checkpoint_freq=10,
        keep_checkpoints_num=3,
        test_mode=True,
    )


if __name__ == "__main__":
    # _test()
    _my_test()
