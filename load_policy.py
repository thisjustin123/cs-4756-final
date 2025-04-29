import datetime
import os
from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel, BasePolicy
from stable_baselines3.sac.policies import MlpPolicy
from imitation.policies import base
import sys

def get_policy(filename: str) -> BaseModel:
  """
  Returns the policy loaded from the given filename
  """
  # Get filename argument
  policy = ActorCriticPolicy.load(filename)
  return policy

def generate_filename(directory: str, type:str="policy") -> str:
  """
  Generates a policy filename.
  """
  # Get the number of existing files in the directory
  existing_files = [f for f in os.listdir(directory)]
  i = len(existing_files)  # This is the ith file
  
  # Get the current date and time, formatted as MMDDHHMM
  timestamp = datetime.datetime.now().strftime("%m%d%H%M")
  
  # Create the filename
  filename = f"{directory}/{type}_{i}_{timestamp}.zip"
  
  return filename
