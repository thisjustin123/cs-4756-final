# Game Plan - BARK Behavior Cloning Warm Start

## OpenAI Gym Integration

BARK-ML integrates directly with `gym` to provide its environments. As such, any libraries that use a `gym` environment to learn, such as Stable Baselines 3, can learn a policy for the environment.

With this in mind, we can use `Imitation` and other similar structures to learn a policy with Behavior Cloning and such, just using this `gym` integration and not too much more from BARK-ML.

## IDM Data Collection

`collect_idm_data.py` is a file which, when run, collects many samples across many merging scenarios (tweakable by the CONSTANTS in the file). These are stored as state, action pairs, in the form of (10,000, 14)--where (12,) is the state shape and (2,) is the action shape; the two are stacked.

## Behavior Cloning

We can run the BehaviorIDMClassic on the ego driver to obtain a list of state action pairs that we later use for behavior cloning.

It is important to use the same observer (state representation) and action representation (2-dim), as well as the same environment. These things are all configurable, so it is important to match the same ones.

With this data, we can run behavior cloning to obtain a policy, and then checkpoint that policy.

## SAC

We then use the BC policy to warm start SAC; likely through Stable Baselines 3, using the `imitation` library on top.

## Notes

- The IDM state/action representation needs to align with that of `"merging-v0"` in order for learning to be successful. A challenge with that is that neither of them are well documented.
  - We need to use an observer of `NearestAgentsObserver` to convert the state to 12-dimensional. But it still might not exactly line up?
