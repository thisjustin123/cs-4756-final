# Copyright (c) 2020 fortiss GmbH
#
# Authors: Julian Bernhard, Klemens Esterle, Patrick Hart and
# Tobias Kessler
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

try:
    import debug_settings
except:
    pass

import datetime
import os
import sys
import time
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.buffered_mp_viewer import BufferedMPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer
from bark.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase
from bark.runtime.runtime import Runtime
from bark.examples.paths import Data
from bark_ml.environments.blueprints.merging.merging import MergingLaneCorridorConfig

from bark.core.world.opendrive import *
from bark.core.world.goal_definition import *
from bark.core.models.behavior import *
from bark.core.commons import SetVerboseLevel
import numpy as np

from data_collect_runtime import DataCollectRuntime

# hyperparams
NUM_SCENARIOS = 500
STEPS_PER_SCENARIO = 200
BEHAVIORS = {
  "idm": BehaviorIDMClassic,
  "idm_lane": BehaviorIDMLaneTracking,
  "lane": BehaviorLaneChangeRuleBased,
  "mobil": BehaviorMobilRuleBased,
}

# parameters
param_server = ParameterServer()

param_server["BehaviorIDMClassic"]["BrakeForLaneEnd"] = True
param_server["BehaviorIDMClassic"]["BrakeForLaneEndEnabledDistance"] = 60.0
param_server["BehaviorIDMClassic"]["BrakeForLaneEndDistanceOffset"] = 30.0
param_server["BehaviorLaneChangeRuleBased"]["MinRemainingLaneCorridorDistance"] = 80.
param_server["BehaviorLaneChangeRuleBased"]["MinVehicleRearDistance"] = 0.
param_server["BehaviorLaneChangeRuleBased"]["MinVehicleFrontDistance"] = 0.
param_server["BehaviorLaneChangeRuleBased"]["TimeKeepingGap"] = 0.
param_server["BehaviorMobilRuleBased"]["Politeness"] = 1.0
param_server["BehaviorIDMClassic"]["DesiredVelocity"] = 10.
param_server["World"]["LateralDifferenceThreshold"] = 0.8

SetVerboseLevel(0)

ego_behavior = BehaviorIDMClassic
behavior_name = "idm"

# Check if user passed in an argument
if len(sys.argv) > 1:
    behavior_name = sys.argv[1].lower()
    if behavior_name in BEHAVIORS:
        ego_behavior = BEHAVIORS[behavior_name]
        print(f"Using behavior: {behavior_name}")
    else:
        print(f"Unknown behavior '{behavior_name}', defaulting to IDM.")

# configure both lanes of the highway. the right lane has one controlled agent
left_lane = MergingLaneCorridorConfig(
      params=param_server,
      road_ids=[0, 1],
      ds_min=7.,
      ds_max=12.,
      min_vel=9.,
      max_vel=11.,
      s_min=5.,
      s_max=45.,
      lane_corridor_id=0,
      controlled_ids=None,
      behavior_model=BehaviorIDMClassic(param_server))
right_lane = MergingLaneCorridorConfig(
  params=param_server,
  road_ids=[0, 1],
  lane_corridor_id=1,
  ds_min=7.,
  ds_max=12.,
  s_min=5.,
  s_max=25.,
  min_vel=9.,
  max_vel=11.,
  controlled_ids=True,
  behavior_model=ego_behavior(param_server))

scenarios = \
  ConfigWithEase(num_scenarios=NUM_SCENARIOS,
                 map_file_name=Data.xodr_data("DR_DEU_Merging_MT_v01_centered"),
                 random_seed=0,
                 params=param_server,
                 lane_corridor_configs=[left_lane, right_lane])

# viewer
viewer = BufferedMPViewer(params=param_server)

sim_step_time = param_server["simulation"]["step_time",
                                           "Step-time used in simulation",
                                           0.1]
sim_real_time_factor = param_server["simulation"]["real_time_factor",
                                                  "execution in real-time or faster",
                                                  1.]

render = True if len(sys.argv) > 2 and sys.argv[2] == "--vis" else False
env = DataCollectRuntime(step_time=sim_step_time,
              viewer=viewer,
              scenario_generator=scenarios,
              render=render,
              maintain_world_history=True)

# Run the scenarios
sa_pair_arrays = []
for i in range(NUM_SCENARIOS):
  env.reset()
  for step in range(STEPS_PER_SCENARIO):
    env.step()
  pairs, state_shape, action_shape = env.get_state_action_pairs()
  sa_pair_arrays.append(pairs)

  if i % 10 == 0:
    print(f"Finished scenario {i}")

# Save all state-action pairs
sa_pairs = np.concatenate(sa_pair_arrays, axis=0)
print("Obtained state-action pairs:", sa_pairs)
print("Shape:", sa_pairs.shape, "with state shape", state_shape, "and action shape", action_shape)
directory = "data"
filename = env.generate_filename(behavior_name=behavior_name, directory=directory, data=sa_pairs)
filepath = os.path.join(directory, filename)
# Save the array to the .npy file
np.save(filepath, sa_pairs)

print(f"Data saved to {filepath}")
