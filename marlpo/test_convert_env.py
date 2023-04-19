from ray.tune.logger import pretty_print
from ray.rllib.env.multi_agent_env import make_multi_agent

# By gym string:
ma_cartpole_cls = make_multi_agent("CartPole-v1") 
# Create a 2 agent multi-agent cartpole.
ma_cartpole = ma_cartpole_cls({"num_agents": 2}) 
obs, info = ma_cartpole.reset() 
print(pretty_print(obs)) 

# By env-maker callable:
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
ma_stateless_cartpole_cls = make_multi_agent( 
   lambda config: StatelessCartPole(config)) 
# Create a 3 agent multi-agent stateless cartpole.
ma_stateless_cartpole = ma_stateless_cartpole_cls({"num_agents": 3}) 
# print(obs) 
print(pretty_print(obs)) 