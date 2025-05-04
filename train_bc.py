import datetime
import gym
import numpy as np
from gym import spaces
# registers bark-ml environments
from imitation.algorithms import bc
from bark_ml.commons.py_spaces import BoundedContinuous
import bark_ml.environments.gym
from imitation.data.types import TransitionsMinimal
import sys
import os

from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner
from stable_baselines3.sac.policies import MlpPolicy

from load_policy import generate_filename

def main(env_name: str, filename: str):
  print(f"Loading data from {filename}")
  data = np.load(filename).astype(np.float32)

  env = gym.make(env_name)

  state_dim = env.observation_space.shape[0]
  
  states = data[:, :state_dim]  # First 12 columns are the state
  actions = data[:, state_dim:]  # Last 2 columns are the action
  infos = np.zeros((data.shape[0], 1))
  transitions = TransitionsMinimal(obs=states, acts=actions, infos=infos)

  bounded_space: BoundedContinuous = env.action_space

  obs_space = spaces.Box(
     low = env.observation_space.low,
     high = env.observation_space.high,
     shape = (env.observation_space.shape[0],),
     dtype = np.float32
  )

  action_space = spaces.Box(
    low = bounded_space.low,
    high = bounded_space.high,
    shape = (bounded_space.n,),
    dtype = np.float32
  )

  bc_trainer = bc.BC(
    observation_space=obs_space,
    action_space=action_space,
    expert_data=transitions,
  )

  save_interval = 1
  TOTAL_EPOCHS = 10

  for epoch in range(TOTAL_EPOCHS):
      bc_trainer.train(n_epochs=1)  # Train for 1 epoch
      if (epoch + 1) % save_interval == 0:
          filename = generate_filename("policies")
          bc_trainer.policy.save(filename)

if __name__ == "__main__":
  # Get filename arg, if it exists
  if len(sys.argv) >= 3:
    env_name = sys.argv[1]
    filename = sys.argv[2]
    
  main(env_name, filename)
    