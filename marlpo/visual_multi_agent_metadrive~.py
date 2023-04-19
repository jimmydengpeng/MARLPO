from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_trainable, register_env

try:
    from env.multi_agent_metadrive import RLlibMultiAgentMetaDrive as MultiAgentMetaDriveEnv
except:
    from .env._multi_agent_metadrive import RLlibMultiAgentMetaDrive as MultiAgentMetaDriveEnv
    


def env_creator(env_config):
    return MultiAgentMetaDriveEnv(env_config)

register_env("MultiAgentMetaDriveEnv", env_creator)




num_agent = 20
config = dict(
            scene='roundabout',
            # scene='intersection',
            use_render=False,
            num_agents=num_agent,
            return_single_space=True,
            # "manual_control": True,
            # crash_done=True,
            # "agent_policy": ManualControllableIDMPolicy
        )


ckp_dir = "/Users/jimmy/ray_results/PPO_MultiAgentMetaDriveEnv_2023-04-10_16-12-roundabout-20agents/checkpoint_000651"
# algo = Algorithm.from_checkpoint(ckp_dir)
ppo_algo = (
        PPOConfig()
        .framework('torch')
        .resources(num_gpus=0)
        .rollouts(
            num_rollout_workers=1,
            create_env_on_local_worker=True,
            # rollout_fragment_length=200
        )
        .training(
            train_batch_size=1024,
            gamma=0.99,
            lr=3e-4,
            sgd_minibatch_size=512,
            num_sgd_iter=5,
            lambda_=0.95,
        )
        .evaluation(
            # evaluation_config={'render',True},
        )
        .multi_agent(
        )
        # .evaluation(
        #     evaluation_interval=2,
        #     evaluation_duration=40,
        #     evaluation_config=dict(env_config=dict(environment_num=200, start_seed=0)),
        #     evaluation_num_workers=1,)
        .environment(env="MultiAgentMetaDriveEnv", render_env=True, env_config=config, disable_env_checking=False)
        .build()
    )

ppo_algo.load_checkpoint(ckp_dir)
worker = ppo_algo.workers.local_worker()
# sample_batch = worker.sample()
# print(sample_batch)
worker.assert_healthy()
# policy = worker.get_policy()
# print(policy)
# worker.preprocessors['default_policy'] = lambda x: x
# print(worker.preprocessors)
# preprocessed = worker.preprocessors[policy_id].transform(ob)
# env = MultiAgentMetaDriveEnv(config)
# obs, info = env.reset()
# print(type(obs))
# for agent_id, ob in obs.items():
#     print(agent_id, ob)
# a = ppo_algo.compute_actions(obs)

res = ppo_algo.evaluate()
print(res)
