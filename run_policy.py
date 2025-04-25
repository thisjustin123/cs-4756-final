import sys

import gym
import numpy as np
from load_policy import get_policy
import bark_ml.environments.gym

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python run_policy.py <policy_file>")
    exit(1)

  policy = get_policy(sys.argv[1])
  env = gym.make("merging-v0")

  MAX_RUNS = 100
  MAX_ITERS = 1000

  i = 0
  r = 0
  total_total_reward = 0
  while r < MAX_RUNS and i < MAX_ITERS:
    obs = env.reset()
    done = False
    total_reward = 0
    r += 1
    while done is False and i < MAX_ITERS:
      prediction = policy.predict(obs)
      action = np.array(prediction[0]).astype(np.float64).reshape(-1, 1)
      obs, reward, done, info = env.step(action)
      total_reward += reward
      i += 1
    total_total_reward += total_reward
    print(f"RUN {r}: ", f"Total reward: {total_reward}", f"Average reward: {total_reward/i}")

  print(f"Final average reward: {total_total_reward/i}")
