from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

config = (
    PPOConfig()
    .framework('torch')
    .resources(num_gpus=0)  
    .rollouts(num_rollout_workers=4)  
    .training(
        train_batch_size=20000, 
        gamma=0.9, 
        lr=3e-4, 
        sgd_minibatch_size=256, 
        num_sgd_iter=10,
        lambda_=0.95,
    )
    .evaluation(
        evaluation_interval=2,
        evaluation_duration=40,
        evaluation_config=dict(env_config=dict(environment_num=200, start_seed=0)),
        evaluation_num_workers=5,)
)

print(pretty_print(config.to_dict())) 
exit()
# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build(env="CartPole-v1")  
res = algo.train()
print(res)