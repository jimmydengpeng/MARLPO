# Create a rollout worker and using it to collect experiences.
import gymnasium as gym
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.algorithms.pg.pg_torch_policy import PGTorchPolicy
from ray.rllib.algorithms.pg.pg_tf_policy import PGTF1Policy
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.tune.logger import pretty_print

worker = RolloutWorker( 
  env_creator=lambda _: gym.make("CartPole-v1"), 
  default_policy_class=PPOTorchPolicy, 
  config=AlgorithmConfig().framework("torch"), 
)

sample_batch = worker.sample()
print(sample_batch)
print(pretty_print(sample_batch.__dict__))
print(sample_batch.agent_steps())
