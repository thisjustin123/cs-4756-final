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
import time
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.buffered_mp_viewer import BufferedMPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer
from bark.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase
from bark.runtime.runtime import Runtime
from bark.examples.paths import Data

from bark.core.world.opendrive import *
from bark.core.world.goal_definition import *
from bark.core.models.behavior import *
from bark.core.commons import SetVerboseLevel
import numpy as np

from data_collect_runtime import DataCollectRuntime

# hyperparams
NUM_SCENARIOS = 100
STEPS_PER_SCENARIO = 100

# parameters
param_server = ParameterServer()

# scenario
class CustomLaneCorridorConfig(LaneCorridorConfig):
  def __init__(self,
               params=None,
               **kwargs):
    super(CustomLaneCorridorConfig, self).__init__(params, **kwargs)
  
  def goal(self, world):
    road_corr = world.map.GetRoadCorridor(
      self._road_ids, XodrDrivingDirection.forward)
    lane_corr = self._road_corridor.lane_corridors[0]
    return GoalDefinitionPolygon(lane_corr.polygon)

param_server["BehaviorIDMClassic"]["BrakeForLaneEnd"] = True
param_server["BehaviorIDMClassic"]["BrakeForLaneEndEnabledDistance"] = 60.0
param_server["BehaviorIDMClassic"]["BrakeForLaneEndDistanceOffset"] = 30.0
param_server["BehaviorLaneChangeRuleBased"]["MinRemainingLaneCorridorDistance"] = 80.
param_server["BehaviorLaneChangeRuleBased"]["MinVehicleRearDistance"] = 0.
param_server["BehaviorLaneChangeRuleBased"]["MinVehicleFrontDistance"] = 0.
param_server["BehaviorLaneChangeRuleBased"]["TimeKeepingGap"] = 0.
param_server["BehaviorMobilRuleBased"]["Politeness"] = 0.0
param_server["BehaviorIDMClassic"]["DesiredVelocity"] = 10.
param_server["World"]["LateralDifferenceThreshold"] = 0.8

SetVerboseLevel(0)

# configure both lanes of the highway. the right lane has one controlled agent
left_lane = CustomLaneCorridorConfig(params=param_server,
                                     lane_corridor_id=0,
                                     road_ids=[0, 1],
                                     behavior_model=BehaviorMobilRuleBased(param_server),
                                     s_min=5.,
                                     s_max=50.)
right_lane = CustomLaneCorridorConfig(params=param_server,
                                      lane_corridor_id=1,
                                      road_ids=[0, 1],
                                      controlled_ids=True,
                                      behavior_model=BehaviorMobilRuleBased(param_server),
                                      s_min=5.,
                                      s_max=20.)

scenarios = \
  ConfigWithEase(num_scenarios=NUM_SCENARIOS,
                 map_file_name=Data.xodr_data("DR_DEU_Merging_MT_v01_shifted"),
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

env = DataCollectRuntime(step_time=sim_step_time,
              viewer=viewer,
              scenario_generator=scenarios,
              render=False,
              maintain_world_history=True)

# Run the scenarios
sa_pair_arrays = []
for i in range(NUM_SCENARIOS):
  env.reset()
  for step in range(STEPS_PER_SCENARIO):
    env.step()
  pairs, state_shape, action_shape = env.get_state_action_pairs()
  sa_pair_arrays.append(pairs)

# Save all state-action pairs
sa_pairs = np.concatenate(sa_pair_arrays, axis=0)
print("Obtained state-action pairs:", sa_pairs)
print("Shape:", sa_pairs.shape, "with state shape", state_shape, "and action shape", action_shape)
directory = "data"
filename = env.generate_filename(directory, sa_pairs)
filepath = os.path.join(directory, filename)
# Save the array to the .npy file
np.save(filepath, sa_pairs)

print(f"Data saved to {filepath}")
