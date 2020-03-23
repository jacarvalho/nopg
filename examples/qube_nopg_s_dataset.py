import math
import gym
from gym.envs import register
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os

from src.mdp.mdp import MDP
from src.policy.policy import Policy
from src.configs.configs import TORCH_DTYPE
from src.nopg.nopg import NOPG
from src.dataset.dataset import Dataset

# Use the GPU if available, or if the memory is insufficient use only the CPU
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

##########################################################################################
# Create the Environment (MDP)
register(
    id='Qube-100-v1',
    entry_point='quanser_robots.qube.qube:Qube',
    max_episode_steps=1500,
    kwargs={'fs': 200.0, 'fs_ctrl': 200.0}
)

env = gym.make('Qube-100-v1')
mdp = MDP(env)

##########################################################################################
# Gather an Off-Policy Dataset
results_dir = '/home/carvalho/Documents/projects/nopg/results/qube/nopgs/'
os.makedirs(results_dir, exist_ok=True)

# Load trajectories from file
filename = '/home/carvalho/Documents/projects/nopg/datasets/qube/15_trajectories.npy'
dataset = Dataset(results_dir=results_dir)
dataset.load_trajectories_from_file(filename, n_trajectories=8)
dataset.update_dataset_internal()
s_band_factor = [15., 15., 15, 15., 1., 1.]
s_n_band_factor = s_band_factor
a_band_factor = [7.]
dataset.kde_bandwidths_internal(s_band_factor=s_band_factor, a_band_factor=a_band_factor,
                                s_n_band_factor=s_n_band_factor)
dataset.plot_data_kde(state_labels=mdp._env.observation_space.labels, action_labels=mdp._env.action_space.labels)


##########################################################################################
# Define the Policy Network

policy_params = {'policy_class': 'stochastic',
                 'neurons': [mdp.s_dim, 60, mdp.a_dim],  # [state_dim, hidden1, ... , hiddenN, action_dim]
                 'activations': [nn.functional.relu],  # one activation function per hidden layer
                 'f_out': [lambda x: 2.5 * torch.tanh(x), lambda x: 0.5 * torch.sigmoid(x)],
                 'device': DEVICE
                 }

policy = Policy(**policy_params).to(device=DEVICE, dtype=TORCH_DTYPE)

##########################################################################################
# Optimize the policy with NOPG

nopg_params = {'initial_states': np.array([env.reset() for _ in range(6)]),  # For multiple initial states
               'gamma': 0.9991,
               'MC_samples_stochastic_policy': 8,
               # 'MC_samples_P': 15,
               # 'sparsify_P': {'kl_max': 0.001, 'kl_interval_k': 20, 'kl_repeat_every_n_iterations': 5}
               }

nopg = NOPG(dataset, policy, **nopg_params)

n_policy_updates = 1500
def optimizer(x): return optim.Adam(x, lr=5e-3)
evaluation_params = {'eval_mdp': mdp,
                     'eval_every_n': 25,
                     'eval_n_episodes': 1,
                     'eval_render': False
                     }

nopg.fit(n_policy_updates=n_policy_updates, optimizer=optimizer, **evaluation_params)
print()
