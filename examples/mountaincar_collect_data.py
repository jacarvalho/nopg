import gym

from src.mdp.mdp import MDP
from src.policy.policy import mountaincar_policy


# Environment
env = gym.make('MountainCarContinuous-v0')
mdp = MDP(env)

# Off-Policy Dataset
sampling_params = {'sampling_type': 'behavioral',
                   'policy': mountaincar_policy,
                   'n_trajectories': 3,
                   'max_ep_transitions': env._max_episode_steps,
                   'render': True
                   }

dataset = mdp.get_samples(**sampling_params)
dataset.update_dataset_internal()

filename = '../../datasets/mountaincar/3_trajectories.npy'
dataset.save_trajectories_to_file(filename)
