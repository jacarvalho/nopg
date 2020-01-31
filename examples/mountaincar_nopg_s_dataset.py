import math
import gym
from gym.envs import register
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from src.mdp.mdp import MDP
from src.policy.policy import Policy, policy_gmm
from src.utils.utils import DEVICE, TORCH_DTYPE, NP_DTYPE
from src.nopg.nopg import NOPG
from src.dataset.dataset import Dataset


##########################################################################################
# Create the Environment (MDP)
env = gym.make('MountainCarContinuous-v0')
mdp = MDP(env)

##########################################################################################
# Gather an Off-Policy Dataset

# Load trajectories from file
filename = '../datasets/mountaincar/10_trajectories.npy'
dataset = Dataset()
dataset.load_trajectories_from_file(filename, n_trajectories=5)
dataset.update_dataset_internal()
s_band_factor = [2., 2.]
s_n_band_factor = s_band_factor
a_band_factor = [50.]
dataset.kde_bandwidths_internal(s_band_factor=s_band_factor, a_band_factor=a_band_factor,
                                s_n_band_factor=s_n_band_factor)

##########################################################################################
# Define the Policy Network

policy_params = {'policy_class': 'stochastic',
                 'neurons': [mdp.s_dim, 50, mdp.a_dim],  # [state_dim, hidden1, ... , hiddenN, action_dim]
                 'activations': [nn.functional.relu],  # one activation function per hidden layer
                 'f_out': [lambda x: 1.0 * torch.tanh(x), lambda x: 1.0 * torch.sigmoid(x)]
                 }

policy = Policy(**policy_params).to(device=DEVICE, dtype=TORCH_DTYPE)

##########################################################################################
# Optimize the policy with NOPG

nopg_params = {'initial_states': np.array([env.reset() for _ in range(15)]),  # For multiple initial states
               'gamma': 0.99,
               'MC_samples_stochastic_policy': 30,
               'sparsify_P': {'kl_max': 0.0001, 'kl_interval_k': 20, 'kl_repeat_every_n_iterations': 200}
               }

nopg = NOPG(dataset, policy, **nopg_params)

n_policy_updates = 1000
def optimizer(x): return optim.Adam(x, lr=1e-3)
evaluation_params = {'eval_mdp': mdp,
                     'eval_every_n': 200,
                     'eval_n_episodes': 1,
                     'eval_render': False
                     }

nopg.fit(n_policy_updates=n_policy_updates, optimizer=optimizer, **evaluation_params)
print()
