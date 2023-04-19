from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

env_id = "CartPole-v1"
# env_id = "LunarLander-v2"
algo = (
    PPOConfig()
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env=env_id, render_env=False)
    .build()
)

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")