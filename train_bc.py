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

if __name__ == "__main__":
  env = gym.make("merging-v0")

  # Get filename arg, if it exists
  if len(sys.argv) >= 2:
    filename = sys.argv[1]
    print(f"Loading data from {filename}")
    data = np.load(filename).astype(np.float32)
  else:
    # Load most recent file in "data" directory
    files = [f for f in os.listdir("data") if f.startswith("idm_data_")]
    filename = files[-1]
    print(f"Loading data from {filename}")
    data = np.load(os.path.join("data", filename)).astype(np.float32)
  
  states = data[:, :12]  # First 12 columns are the state
  actions = data[:, 12:]  # Last 2 columns are the action
  infos = np.zeros((data.shape[0], 1))
  transitions = TransitionsMinimal(obs=states, acts=actions, infos=infos)

  bounded_space: BoundedContinuous = env.action_space

  action_space = spaces.Box(
    low = bounded_space.low,
    high = bounded_space.high,
    shape = (bounded_space.n,),
    dtype = np.float32
  )

  bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=action_space,
    expert_data=transitions,
  )

  save_interval = 2
  TOTAL_EPOCHS = 10

  for epoch in range(TOTAL_EPOCHS):
      bc_trainer.train(n_epochs=1)  # Train for 1 epoch
      if (epoch + 1) % save_interval == 0:
          filename = generate_filename("policies")
          bc_trainer.policy.save(filename)
