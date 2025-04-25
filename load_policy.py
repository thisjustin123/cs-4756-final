import datetime
import os
from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel
import sys

def get_policy(filename: str) -> BaseModel:
  # Get filename argument
  filename = sys.argv[1]
  policy = ActorCriticPolicy.load(filename)
  return policy

def generate_filename(directory: str, type:str="policy") -> str:
    # Get the number of existing files in the directory
    existing_files = [f for f in os.listdir(directory) if f.startswith("idm_data_")]
    i = len(existing_files)  # This is the ith file
    
    # Get the current date and time, formatted as MMDDHHMM
    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    
    # Create the filename
    filename = f"{directory}/{type}_{i}_{timestamp}.npy"
    
    return filename
