import datetime
import sys
import gym
import numpy as np

from classes_and_examples.data_collect_env import *

def generate_filename(behavior_name:str, env_name:str, directory: str, data: np.ndarray) -> str:
    # Get the number of existing files in the directory
    existing_files = [f for f in os.listdir(directory)]
    i = len(existing_files)  # This is the ith file
    
    # Get the current date and time, formatted as MMDDHHMM
    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    
    # Get the number of data points (shape[0])
    num_data_points = data.shape[0]
    
    # Create the filename
    filename = f"{behavior_name}_{env_name}_{i}_{timestamp}_{num_data_points}.npy"
    
    return filename

def main(env_name: str, behavior_type: str, render: bool):
  env = gym.make(env_name, behavior_type=behavior_type, render=render)
  obs = env.reset()

  state_list = []
  action_list = []
  for i in range(100000):
    obs, reward, done, info = env.step(None)
    action = env.get_last_action()

    state_list.append(obs)
    action_list.append(action)

    if i % 1000 == 0:
      print(f"Finished step {i}")

    if done:
      obs = env.reset()

  sa_pairs = np.column_stack((state_list, action_list))
  directory = "data"
  filename = generate_filename(behavior_name=behavior_type, directory=directory, data=sa_pairs, env_name=env_name)
  filepath = os.path.join(directory, filename)
  # Save the array to the .npy file
  np.save(filepath, sa_pairs)

  print(f"Data saved to {filepath}")

if __name__ == '__main__':
  # sys args:
  # 1: env name
  # 2: behavior type
  # 3: render, if it exists as --vis
  env_name = sys.argv[1]
  behavior_type = sys.argv[2]
  render = True if '--vis' in sys.argv else False

  main(env_name=env_name, behavior_type=behavior_type, render=render)
  
