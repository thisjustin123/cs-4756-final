import os
import sys
import gym
import numpy as np


from gym.spaces import Box

from load_policy import generate_filename, get_policy
from bark_ml.commons.py_spaces import BoundedContinuous

from stable_baselines3 import SAC
from stable_baselines3.common.policies import ActorCriticPolicy
import bark_ml.environments.gym
from imitation.policies import base


if __name__ == "__main__":
  env = gym.make("merging-v0")

  # Get filename arg, if it exists
  if len(sys.argv) >= 2:
    filename = sys.argv[1]
    print(f"Warm starting policy from {filename}")
    warm_policy = get_policy(filename)
  else:
    print("NOT Warm starting.")
    warm_policy = None

  bounded_space: BoundedContinuous = env.action_space

  action_space = Box(
    low = bounded_space.low,
    high = bounded_space.high,
    shape = (bounded_space.n,),
    dtype = np.float32
  )

  env.action_space = action_space
  
  # Train
  print(warm_policy.__class__)
  sac_trainer = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=1,
  )

  sac_trainer.actor.load_state_dict(warm_policy.state_dict())

  sac_trainer.learn(total_timesteps=1000)

  filename = generate_filename(directory="models", type=f"sac")
  sac_trainer.save(filename)

  
