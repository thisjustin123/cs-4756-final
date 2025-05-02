import os
import sys

import gym
import numpy as np
import matplotlib.pyplot as plt
from load_policy import get_policy
from stable_baselines3.common.policies import BaseModel
import bark_ml.environments.gym

from typing import List

# Should use MAX_ITERS since trajectories vary significantly in length.
MAX_ITERS = 1000
MAX_RUNS = MAX_ITERS

def run_policy(policy: BaseModel, env: gym.Env):
  """
  Runs the given policy on the given environment for MAX_ITERS.

  Args:
    policy: The policy to run.
    env: The environment to run the policy on.

  Returns:
    A tuple. The first element is the average reward over MAX_RUNS, the second element is the total reward. The third is the safety rate.
  """
  total_iters = 0
  r = 0
  total_average_reward = 0
  total_reward = 0
  crashes = 0
  while r < MAX_RUNS and total_iters < MAX_ITERS:
    obs = env.reset()
    done = False
    run_reward = 0
    r += 1
    i = 0
    while done is False and total_iters < MAX_ITERS:
      prediction = policy.predict(obs)
      action = np.array(prediction[0]).astype(np.float64).reshape(-1, 1)
      obs, reward, done, info = env.step(action)

      run_reward += reward
      total_reward += reward
      if done and reward <= -0.9:
        crashes += 1
      
      total_iters += 1
      i += 1

    run_average_reward = run_reward/i
    print(f"RUN {r}: ", f"Total reward: {run_reward}", f"Average reward: {run_average_reward}")
    total_average_reward += run_average_reward

  print(f"Final average reward: {total_average_reward/r}, Final total reward: {total_reward}")
  return total_average_reward/r, total_reward, (r - crashes)/r

def format_policy_name(policy_name: str):
  """
  Returns a nicely formatted version of the policy name.
  """
  names = {
    "idm_lane": "IDM Lane Tracking",
    "idm": "IDM",
    "mobil": "MOBIL",
    "lane": "Lane Change",
    "cold": "Cold Start",
  }

  for key in names:
    if key in policy_name:
      return names[key]

  return policy_name.replace("_", " ").replace("ppo", "").replace("bc", "").replace(".zip", "")

def plot_bar(title: str, ylabel: str, labels: List[str], values: List[float], colors: List[str]):
  bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2, width=1.0)
  for bar, label in zip(bars, labels):
        bar.set_label(label)
  plt.legend(title="Warm Start Policy")
  plt.ylabel(ylabel)
  plt.title(title)
  plt.xticks(rotation=45) 
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python run_policy.py <env_name> <policy_file>")
    exit(1)

  env_name = sys.argv[1]
  env = gym.make(env_name)
  avg_rewards = []  
  total_rewards = []
  safety_rates = []

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
    
    avg_reward, total_reward, safety_rate = run_policy(policy, env)
    avg_rewards.append(avg_reward)
    total_rewards.append(total_reward)
    safety_rates.append(safety_rate)
    
  # Plot and display
  colors=["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]
  labels=[format_policy_name(p) for p in policies]
  plot_bar(title="Average Reward by PPO Warm Start Method", ylabel="Average Reward", labels=labels, values=avg_rewards, colors=colors)

  plot_bar(title="Total Reward by PPO Warm Start Method", ylabel="Total Reward", labels=labels, values=total_rewards, colors=colors)

  plot_bar(title="Safety Rate by PPO Warm Start Method", ylabel="Safety Rate", labels=labels, values=safety_rates, colors=colors)



  
