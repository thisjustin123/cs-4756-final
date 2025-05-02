import os
import sys

import gym
import numpy as np
import matplotlib.pyplot as plt
from load_policy import get_policy
from stable_baselines3.common.policies import BaseModel
import bark_ml.environments.gym

MAX_RUNS = 500
MAX_ITERS = 10000

def run_policy(policy: BaseModel, env: gym.Env):
  total_iters = 0
  r = 0
  total_average_reward = 0
  while r < MAX_RUNS and total_iters < MAX_ITERS:
    obs = env.reset()
    done = False
    total_reward = 0
    r += 1
    i = 0
    while done is False and total_iters < MAX_ITERS:
      prediction = policy.predict(obs)
      action = np.array(prediction[0]).astype(np.float64).reshape(-1, 1)
      obs, reward, done, info = env.step(action)
      total_reward += reward
      total_iters += 1
      i += 1

    average_reward = total_reward/i
    print(f"RUN {r}: ", f"Total reward: {total_reward}", f"Average reward: {average_reward}")
    total_average_reward += average_reward
    

  print(f"Final average reward: {total_average_reward/r}")
  return total_average_reward/r

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python run_policy.py <env_name> <policy_file>")
    exit(1)

  env_name = sys.argv[1]
  env = gym.make(env_name)
  rewards = []  

  # If given a directory, run all `.zip` policies in the immediate directory.
  if len(sys.argv) == 3 and os.path.isdir(sys.argv[2]):
    policies = [f for f in os.listdir(sys.argv[2]) if f.endswith(".zip")]
    # Add the directory to the policies
    policies = [os.path.join(sys.argv[2], policy) for policy in policies]
  else:
    # Add each policy individually to run
    policies = sys.argv[2:]

  # Run all policies.
  for arg in policies:
    print(f"Running policy {arg}")
    policy = get_policy(arg)
    
    reward = run_policy(policy, env)
    rewards.append(reward)
    
  # Plot and display
  plt.plot(rewards)
  plt.show()

  
