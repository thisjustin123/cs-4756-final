# Method

All code for our methodology is available at: [`thisjustin123/cs-4756-final`](https://github.com/thisjustin123/cs-4756-final) on GitHub.

For a quick rundown of each script's usage and set of accepted inputs:

* `python collect_data_gym.py <env_name> <behavior_type>` - Runs data collection on the given data collection environment (defined in [`data_collect_env.py`](TODO)) with the given expert behavior type (`"idm"`, `"idm_lane"`, `lane`, or `mobil`). Outputs an `.npy` file with the state action pairs to the `data/` directory.

* `python train_bc.py <env_name> <data filepath>` - Trains BC policy snapshots on the given environment (`highway-v0` or `merging-v0`) with the given expert data `.npy` file. Outputs 10 `.zip` policy snapshots to the `policies/` directory. 

* `python run_policy.py <env_name> <directory or several filepaths> <optional: iters=X>` - Evaluates the policies in the immediate directory on the given environment for a given number of iterations (3000 default). Displays 3 bar graphs for evaluation metrics on each policy: Average Reward, Total Reward, and Safety Rate.

* `python train_ppo.py <env_name> <optional: policy filepath>` - Trains a PPO model on the given environment with the given policy `.zip` as a warm start. If not given a warm start filepath, the PPO will be trained from a cold start. Outputs 10 model `.zip` snapshots to the `models/` directory.

We use these scripts in our methodology across multiple scripts and code snippets, organized into several steps.

For each environment:

1) [`collect_data_gym.py`](TODO) is run to collect data points (100,000 by default) on a given expert behavior model--either IDM, IDM Lane, Lane Change, or Mobil. As part of our implementation, we added a [`data_collect_env.py`](TODO) that exactly mimics the gym environments `merging-v0` and `highway-v0`, except just using the behavior models for the experts we listed above (instead of taking in an `action` per step).

For each expert behavior model (i.e. IDM, IDM Lane, etc.):

2) [`train_bc.py`](TODO) is run to train a Behavior Cloning (BC) model on each set of expert behavior model data. To (later) account for overfitting, we train BC on 10 epochs, saving a snapshot of the model each epoch. 

3) [`run_policy.py`](TODO) evaluates each behavior cloning snapshot and produces a graph that indicates which BC snapshot produces the greatest average reward across its trajectories. We hand select the BC policy that produces the best average reward.

4) [`train_ppo.py`](TODO) uses the trained BC policy to warm start its training for PPO. We then follow a similar method of training for 100,000 total timesteps, saving a snapshot each 10,000 timesteps to later account for overfitting.

5) [`run_policy.py`](TODO) is run again on the PPO snapshots to hand select the best model of the 10 snapshots.

    * Note: For a cold start, we omit step 1 and 2, and just run `train_ppo.py` with no warm start policy.

Finally...

6) Once all PPO models are acquired, [`run_policy.py`](TODO) generates figures comparing all final PPO policies across IDM, IDM Lane, Lane Change, Mobil, and Cold Start. It creates three figures: one for each evaluation metric (Average Reward, Total Reward, and Safety Rate).
