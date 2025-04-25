import gym
import numpy as np
# registers bark-ml environments
import bark_ml.environments.gym

from lane_corridor import CustomLaneCorridorConfig  # pylint: disable=unused-import

from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner

env = gym.make("merging-v0")

initial_state = env.reset()
done = False
while done is False:
  action = np.array([0., 0.]) # acceleration and steering-rate
  observed_state, reward, done, info = env.step(action)
  print(f"Observed state: {observed_state}, Action: {action}, "
        f"Reward: {reward}, Done: {done}.")

left_lane = CustomLaneCorridorConfig(params=param_server,
                                     lane_corridor_id=0,
                                     road_ids=[0, 1],
                                     behavior_model=BehaviorMobilRuleBased(param_server),
                                     s_min=0.,
                                     s_max=50.)
right_lane = CustomLaneCorridorConfig(params=param_server,
                                      lane_corridor_id=1,
                                      road_ids=[0, 1],
                                      controlled_ids=True,
                                      behavior_model=BehaviorMobilRuleBased(param_server),
                                      s_min=0.,
                                      s_max=20.)

scenarios = \
  ConfigWithEase(num_scenarios=3,
                 map_file_name="bark/runtime/tests/data/DR_DEU_Merging_MT_v01_shifted.xodr",
                 random_seed=0,
                 params=param_server,
                 lane_corridor_configs=[left_lane, right_lane])
