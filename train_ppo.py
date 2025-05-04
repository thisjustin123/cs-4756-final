import os
import sys
import gym
import numpy as np


from gym.spaces import Box

from load_policy import generate_filename, get_policy
from bark_ml.commons.py_spaces import BoundedContinuous

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import bark_ml.environments.gym
from imitation.policies import base

from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel, BasePolicy

def main(warm_policy: BaseModel, env_name: str):
  env = gym.make(env_name)

  bounded_space: BoundedContinuous = env.action_space

  obs_space = Box(
     low = env.observation_space.low,
     high = env.observation_space.high,
     shape = (env.observation_space.shape[0],),
     dtype = np.float32
  )

  action_space = Box(
    low = bounded_space.low,
    high = bounded_space.high,
    shape = (bounded_space.n,),
    dtype = np.float32
  )

  env.action_space = action_space
  env.observation_space = obs_space
  
  policy_kwargs = dict(net_arch=[32,32])
  ppo_trainer = PPO(
    policy="MlpPolicy",
    policy_kwargs=policy_kwargs,
    env=env,
    verbose=1,
  )

  if warm_policy is not None:
    ppo_trainer.policy.load_state_dict(warm_policy.state_dict())

  total_steps = 100000
  steps_per_iteration = 10000
  num_iterations = total_steps // steps_per_iteration

  for i in range(num_iterations):
      ppo_trainer.learn(total_timesteps=steps_per_iteration)
      
      filename = generate_filename(directory="models", type=f"ppo_step{i+1}")
      ppo_trainer.save(filename)
      print(f"Policy saved to {filename}")

if __name__ == "__main__":
  # Get filename arg, if it exists
  if len(sys.argv) >= 3:
    filename = sys.argv[2]
    print(f"Warm starting policy from {filename}")
    warm_policy = get_policy(filename)
  else:
    print("NOT Warm starting.")
    warm_policy = None

  main(warm_policy=warm_policy, env_name=sys.argv[1])
