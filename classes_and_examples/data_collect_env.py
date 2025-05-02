"""
Gym environments that collect data for training.
"""

import gym
import os
from gym.envs.registration import register

from bark.runtime.commons.parameters import ParameterServer

from bark_ml.environments.blueprints.highway.highway import \
  ContinuousHighwayBlueprint, DiscreteHighwayBlueprint
from bark_ml.environments.blueprints.merging.merging import \
  ContinuousMergingBlueprint, DiscreteMergingBlueprint
from bark_ml.environments.blueprints.single_lane.single_lane import \
  ContinuousSingleLaneBlueprint
from bark_ml.environments.blueprints.intersection.intersection import \
  ContinuousIntersectionBlueprint, DiscreteIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime

from bark.core.models.behavior import *

# hyperparams
BEHAVIORS = {
  "idm": BehaviorIDMClassic, # type: ignore
  "idm_lane": BehaviorIDMLaneTracking, # type: ignore
  "lane": BehaviorLaneChangeRuleBased, # type: ignore
  "mobil": BehaviorMobilRuleBased, # type: ignore
}

class GymDataCollectRuntime(SingleAgentRuntime):
  """Runtime that collects data for training."""

  def __init__(self,
               blueprint=None,
               ml_behavior=None,
               observer=None,
               evaluator=None,
               step_time=None,
               viewer=None,
               scenario_generator=None,
               render=False):
    
    # Just use the SingleAgentRuntime constructor.
    super().__init__(blueprint=blueprint,
                     ml_behavior=ml_behavior,
                     observer=observer,
                     evaluator=evaluator,
                     step_time=step_time,
                     viewer=viewer,
                     scenario_generator=scenario_generator,
                     render=render)
    
  def step(self, _):
    """
    For data collection purposes, `action` argument is ignored.

    Args:
      action: Ignored

    Returns:
      state, reward, done, info
    """
    # set actions
    eval_id = self._scenario._eval_agent_ids[0]
    # DO NOT Run this code; this would plan for the ego vehicle.
    # if eval_id in self._world.agents:
    #   self._world.agents[eval_id].behavior_model.ActionToBehavior(action)

    # step and observe
    self._world.Step(self._step_time)
    observed_world = self._world.Observe([eval_id])

    if len(observed_world) > 0:
      observed_world = observed_world[0]
    else:
      raise Exception('No world instance available.')

    # observe and evaluate
    observed_next_state = self._observer.Observe(observed_world)
    action = self._world.agents[eval_id].behavior_model.GetLastAction()
    reward, done, info = self._evaluator.Evaluate(
      observed_world=observed_world,
      action=action)

    # render
    if self._render:
      self.render()

    return observed_next_state, reward, done, info
  
  def get_last_action(self):
    eval_id = self._scenario._eval_agent_ids[0]
    return self._world.agents[eval_id].behavior_model.GetLastAction()

# merging
class DataCollectMerging(GymDataCollectRuntime, gym.Env):
  """Merging scenario with continuous behavior model.

  Behavior model takes the steering-rate and acceleration.
  """

  def __init__(self,
               behavior_type='idm',
               render=False):
    params = ParameterServer(filename= \
      os.path.join(os.path.dirname(__file__),
      "visualization_params.json"))
    cont_merging_bp = ContinuousMergingBlueprint(params)
    GymDataCollectRuntime.__init__(self,
      blueprint=cont_merging_bp, render=render, ml_behavior=BEHAVIORS[behavior_type](params))
    
class DataCollectHighway(GymDataCollectRuntime, gym.Env):
  """Highway scenario with continuous behavior model.

  Behavior model takes the steering-rate and acceleration.
  """
  def __init__(self,
               behavior_type='idm',
               render=False):
    params = ParameterServer(filename=
      os.path.join(os.path.dirname(__file__),
      "visualization_params.json"))
    cont_highway_bp = ContinuousHighwayBlueprint(params)
    GymDataCollectRuntime.__init__(self,
      blueprint=cont_highway_bp, render=render, ml_behavior=BEHAVIORS[behavior_type](params))

register(
  id='merging-data-collect',
  entry_point='classes_and_examples.data_collect_env:DataCollectMerging'
)

register(
  id='highway-data-collect',
  entry_point='classes_and_examples.data_collect_env:DataCollectHighway'
)
