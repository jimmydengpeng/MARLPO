import gymnasium as gym
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.algorithms.pg.pg_tf_policy import PGTF1Policy
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

from env.metadrive_env import RLlibMetaDriveEnv as MetaDriveEnv
from env.multi_agent_metadrive import RLlibMultiAgentMetaDrive as MetaDriveEnv
# from env.metadrive_ import SubMetaDriveEnv as MetaDriveEnv

config = dict(
        use_render=False,
        manual_control=True,
        traffic_density=0.1,
        environment_num=100,
        random_agent_model=True,
        random_lane_width=True,
        random_lane_num=True,
        map=4,  # seven block
        start_seed=42
)
def env_creator(config):
    return MetaDriveEnv(config)

# env = env_creator()

worker = RolloutWorker( 
  env_creator=env_creator, 
  default_policy_class=PPOTorchPolicy, 
  config=AlgorithmConfig()
    .framework("torch")
    # .rollouts(num_rollout_workers=1)
    # .resources(num_gpus=0)
)

# env.close()

sample_batch = worker.sample()
print(sample_batch)
print(pretty_print(sample_batch.__dict__))
print(sample_batch.agent_steps())

