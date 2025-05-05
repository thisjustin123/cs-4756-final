import os
import sys

import gym
import numpy as np
import matplotlib.pyplot as plt
from classes.load_policy import get_policy
from stable_baselines3.common.policies import BaseModel
import bark_ml.environments.gym

from typing import List

# Should use MAX_ITERS since trajectories vary significantly in length.
DEFAULT_ITERS = 3000

def run_policy(policy: BaseModel, env: gym.Env, iters: int):
  """
  Runs the given policy on the given environment for MAX_ITERS.

  Args:
    policy: The policy to run.
    env: The environment to run the policy on.

  Returns:
    A tuple. The first element is the average reward over MAX_RUNS, the second element is the total reward. The third is the safety rate.
  """
  MAX_RUNS = iters

  total_iters = 0
  r = 0
  total_average_reward = 0
  total_reward = 0
  crashes = 0
  while r < MAX_RUNS and total_iters < iters:
    obs = env.reset()
    done = False
    run_reward = 0
    r += 1
    i = 0
    while done is False and total_iters < iters:
      prediction = policy.predict(obs)
      action = np.array(prediction[0]).astype(np.float64).reshape(-1, 1)
      obs, reward, done, info = env.step(action)

      run_reward += reward
      total_reward += reward
      if info.get("collision", False):
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
    "idm_lane": "IDM Lane\nTracking",
    "idm": "IDM",
    "mobil": "MOBIL",
    "lane": "Lane\nChange",
    "cold": "Cold\nStart",
  }

  for key in names:
    if key in policy_name:
      return names[key]

  return policy_name.replace("_", " ").replace("ppo", "").replace("bc", "").replace(".zip", "")

def plot_bar(title: str, xlabel: str, ylabel: str, labels: List[str], values: List[float], colors: List[str]):
  bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2, width=0.7)
  for bar, label in zip(bars, labels):
        bar.set_label(label)
  plt.xlabel(xlabel, fontweight='bold', fontsize=12, labelpad=14)
  plt.ylabel(ylabel, fontweight='bold', fontsize=12, labelpad=10)
  plt.title(title, fontweight='bold', fontsize=14, pad=14)
  plt.grid(axis='y', linestyle='--', alpha=0.5)
  plt.xticks(fontsize=10)
  plt.yticks(fontsize=10)
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python run_policy.py <env_name> <policy_file> <optional: iters=XXX>")
    exit(1)

  env_name = sys.argv[1]
  env = gym.make(env_name)
  avg_rewards = []  
  total_rewards = []
  safety_rates = []

  # If given a directory, run all `.zip` policies in the immediate directory.
  if os.path.isdir(sys.argv[2]):
    policies = [f for f in os.listdir(sys.argv[2]) if f.endswith(".zip")]
    # Add the directory to the policies
    policies = [os.path.join(sys.argv[2], policy) for policy in policies]
  else:
    # Add each policy individually to run
    policies = sys.argv[2:]

  iters = DEFAULT_ITERS
  for i in range(3, len(sys.argv)):
    if sys.argv[i].startswith("iters="):
      iters = int(sys.argv[i].split("=")[1])
      print("found iters")
      break    

  # Run all policies.
  for arg in policies:
    print(f"Running policy {arg}")
    policy = get_policy(arg)
    
    avg_reward, total_reward, safety_rate = run_policy(policy, env, iters)
    avg_rewards.append(avg_reward)
    total_rewards.append(total_reward)
    safety_rates.append(safety_rate)
    
  # Plot and display
  colors=["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]
  labels=[format_policy_name(p) for p in policies]
  plot_bar(title=f"Average Reward by PPO Warm Start Method\n({env_name})", xlabel = "Warm Start Policy", ylabel="Average Reward", labels=labels, values=avg_rewards, colors=colors)

  plot_bar(title=f"Total Reward by PPO Warm Start Method\n({env_name})", xlabel = "Warm Start Policy", ylabel="Total Reward", labels=labels, values=total_rewards, colors=colors)

  plot_bar(title=f"Safety Rate by PPO Warm Start Method\n({env_name})", xlabel = "Warm Start Policy", ylabel="Safety Rate", labels=labels, values=safety_rates, colors=colors)



  
