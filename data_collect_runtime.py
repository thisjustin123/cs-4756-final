import datetime
import logging
import os

import numpy as np
import pandas as pd

from bark.core.world.opendrive import *
from bark.core.world import *
from bark.core.geometry import *
from bark.runtime.runtime import Runtime
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver

# import type annotations
from typing import Tuple
import numpy.typing as npt

class DataCollectRuntime(Runtime):

  def __init__(self,
               step_time,
               viewer,
               scenario_generator=None,
               render=False,
               maintain_world_history=False):
    self._params = ParameterServer()
    Runtime.__init__(self, step_time, viewer, scenario_generator, render, maintain_world_history)
    self.observer = NearestAgentsObserver(self._params)
    self.observed_states = []

  def observe(self):
    observed_world = ObservedWorld(self._world, 0)
    return self.observer.Observe(observed_world)
    
  
  def step(self):
    super().step()

    self.observed_states.append(self.observe())

  def reset(self):
    super().reset()
    self.observed_states = []

  def get_state_action_pairs(self) -> Tuple[npt.NDArray[np.float32],np.float32, np.float32]:
    """
    Returns a numpy array of state-action pairs. The shape of the state and action are given as the second and third tuple elements returned.
    
    The first column is the state, the second the action
    """
    state_list = self.observed_states.copy()
    action_list = []
    for w in self._world_history[1:]:
        eval_agent = w.GetAgent(self._scenario._eval_agent_ids[0])
        action_list.append(eval_agent.behavior_model.GetLastAction())

    combined = np.column_stack((state_list, action_list)).astype(np.float32)
    return combined, state_list[0].shape, action_list[0].shape

  def generate_filename(self, directory: str, data: np.ndarray) -> str:
    # Get the number of existing files in the directory
    existing_files = [f for f in os.listdir(directory) if f.startswith("idm_data_")]
    i = len(existing_files)  # This is the ith file
    
    # Get the current date and time, formatted as MMDDHHMM
    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    
    # Get the number of data points (shape[0])
    num_data_points = data.shape[0]
    
    # Create the filename
    filename = f"idm_data_{i}_{timestamp}_{num_data_points}.npy"
    
    return filename
