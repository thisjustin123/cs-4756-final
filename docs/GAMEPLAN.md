# Game Plan - BARK Behavior Cloning Warm Start

## OpenAI Gym Integration

BARK-ML integrates directly with `gym` to provide its environments. As such, any libraries that use a `gym` environment to learn, such as Stable Baselines 3, can learn a policy for the environment.

With this in mind, we can use `Imitation` and other similar structures to learn a policy with Behavior Cloning and such, just using this `gym` integration and not too much more from BARK-ML.

## Expert Data Collection

`collect_data_merging.py` is a file which, when run, collects many samples across many merging scenarios (tweakable by the CONSTANTS in the file). These are stored as state, action pairs, in the form of (10,000, 14)--where (12,) is the state shape and (2,) is the action shape; the two are stacked.

## Behavior Cloning

We can run the `BehaviorMobilRuleBased` and other policies on the ego driver to obtain a list of state action pairs that we later use for behavior cloning. It performs well; better than `BehaviorLaneChangeRuleBased` which seems to just cause collisions.

It is important to use the same observer (state representation) and action representation (2-dim), as well as the same environment. These things are all configurable, so it is important to match the same ones.

With this data, we can run behavior cloning to obtain a policy, and then checkpoint that policy.

## PPO

We then use the BC policy to warm start PPO; likely through Stable Baselines 3, using the `imitation` library on top.

## Notes

- The IDM state/action representation needs to align with that of `"merging-v0"` in order for learning to be successful. A challenge with that is that neither of them are well documented.
  - We need to use an observer of `NearestAgentsObserver` to convert the state to 12-dimensional. But it still might not exactly line up?

# Feedback
```
I think this is a decent start, but it is a bit simplistic and could use some more development.

From a first glance, I don't think the INTERACTION dataset will be useful. Just from the videos shown on the website, it seems to be a different scale of problem with many more agents than the Bark-ML. Processing that data into something used for Bark-ML seems like it would take a long time, and still not be useful. I would recommend hand-labeling or finding a different dataset specifically for Bark-ML.

Second your hypothesis seems to be saying that an autonomous driving expert BC > human labeled BC > no BC. This is a bit simplistic especially since you don't specify any metrics that you will use to test this claim. To add complexity to this project I would recommend having many different classes of evaluation metrics that you could hypothesis and test across. One could maybe be that there will be a difference in amount of data you need to obtain good performance in human BC versus IDM BC.

Another method that might add complexity is to test across more than 1 environment. Another method is to choose an evaluation metric and try to figure out a method of optimizing your BC data collection to outperform the IDM BC on that evaluation metric. Overall, the environment is fine, and IDM expert is probably ok as well, just try to have a few more variables you can test and hypothesis across.
```

### TLDR:
> I don't think the INTERACTION dataset will be useful.
* Don't use INTERACTION Dataset. However, collecting human data seems ass. I think instead we can collect data with some different (complex) behavior model BC, IDM BC, vs. NO bc.

> To add complexity to this project I would recommend having many different classes of evaluation metrics that you could hypothesis and test across.
* We should add multiple specific **evaluation metrics**.

> Another method that might add complexity is to test across more than 1 environment. 

* We should test across `highway-v0` and `merging-v0`.

# Evaluation Metrics & Complexity

We add complexity to our project with the following:

## BC Data

We will collect data from the following policies for warm start:
* `BehaviorMobilRuleBased` (tracking and merging)
* `BehaviorIDMLaneTracking` (lane track; no merging)
* `BehaviorLaneChangeRuleBased` (merging)
* `BehaviorIDMClassic` (simple line movement)
* (No warm start)

## Evaluation Metrics

We run each policy for 10,000 timesteps (resetting whenever a trajectory fails) in order to evaluate / gather the following metrics. This is because our trajectories vary significantly in length, for a very poor policy may repeatedly crash immediately whereas a good policy may take many iterations to crash, or the scene may simply terminate in time.

### Total Reward

The Total Reward is the sum of all timesteps' rewards across all trajectories. We use this as a simple evaluation metric to see which policy accumulates the most rewards across all timesteps. 

However, this may unfairly bias a car that crashes immediately (-1) every step rather than one that tries a bit, gets lightly punished with negative rewards, then crashes. As such, we present our next metric...

### Average Reward (AAR)

The Average reward of a trajectory is sum of the rewards of each timestep, divided by the number of timesteps of that run.

The Average Reward METRIC is the Average of the Average Rewards, averaged across the total number of runs. In other words, it is the average of the average rewards across each trajectory.

We use this as a metric since the total number of timesteps itself serves as a metric for autonomous driving success, given that we observe that the driver almost always crashes (to get a `-1` reward).

### Safety Rate

The majority of runs on poorly trained algorithms cause a crash and a steeply negative reward. As such, using the number of runs in which an algorithm *does not cause a crash* is useful in determining how safe our algorithm is for the road. 

We define safety rate as the proportion of runs in which the policy does *not* cause the car to crash.

## Environments

We will run the entire set of metrics above on both:
* `merging-v0` 
* `highway-v0` 

expecting better results on `highway-v0` since it overall is a simpler problem to solve that more or less requires just driving in line versus performing a tight merge.

BARK-ML also provides `intersection-v0` but that seems way too difficult to learn for.

## Figures

We will have the following figures, PER environment:

* 5-bar graph, one for each evaluation metric (3 graphs total)

## Hypothesis

### `merging-v0`

For merging, we expect the warm start from algorithms that have merging logic built in to outperform the algorithms meant for driving in a single lane in any reward metrics, but to be worse in safety rate since the added merging logic learned may cause additional collisions as opposed to just trying to stay in one lane.

### `highway-v0`

For the highway, we expect merging to cause unnecessary collisions, and as such expect the `idm-lane` BC warm start to significantly outperform all the other algorithms on all metrics. This is because merging is NOT required in the highway scenario, and an algorithm which stays in the same lane will create the steadiest and safest driving behavior to learn from.

# BARK-ML Gym rewrite (DONE) - `data_collect_env.py`

I don't want to have to pray that my constructed environment somehow matches up with their environment. As such, I will try to **create a gym environment / bark ML runtime subclass** that allows me to run a behavior as the expert. I don't think it is too difficult...

This requires that I can construct a `BehaviorIDMClassic()` and pass it in. I'm not completely sure if I can. But IF SO, then the procedure is more or less as follows:

1. Construct an expert behavior and pass it in to my wrapper class.
2. The wrapper class will construct the given gym environment and obtain the observed world at each step.
3. The wrapper will implement `.step()` by simply using the behavior model's .Plan(...) to evaluate the next action.
4. States will then be read from the BARK-ML state and actions.
5. `.step()` will return the state and action pairs.

This way, I do not have to GUESS at which parameters cause what behaviors.

Also, I can probably then ignore data that causes a crash, since that's not very expert. But I'm not sure the best way to do that. Or if I should.

Useful methods:

* `behavior_model.GetLastAction()` -- Returns the last action given by the trajectory of `.Plan(...)`
* `behavior_model.Plan()` -- Plans a new trajectory on the observed world.
