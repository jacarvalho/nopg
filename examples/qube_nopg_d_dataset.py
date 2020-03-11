import math
import gym
from gym.envs import register
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os

from src.mdp.mdp import MDP
from src.policy.policy import Policy, policy_gmm
from src.utils.utils import DEVICE, TORCH_DTYPE, NP_DTYPE
from src.nopg.nopg import NOPG
from src.dataset.dataset import Dataset


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
results_dir = '/home/carvalho/Documents/projects/nopg/results/qube/'
os.makedirs(results_dir, exist_ok=True)

# Load trajectories from file
filename = '../datasets/qube/10_trajectories.npy'
dataset = Dataset(results_dir=results_dir)
dataset.load_trajectories_from_file(filename, n_trajectories=10)
dataset.update_dataset_internal()
s_band_factor = [10., 10., 10., 10., 10., 10.]
s_n_band_factor = s_band_factor
a_band_factor = [20.]
dataset.kde_bandwidths_internal(s_band_factor=s_band_factor, a_band_factor=a_band_factor,
                                s_n_band_factor=s_n_band_factor)
dataset.plot_data_kde(state_labels=mdp._env.observation_space.labels, action_labels=mdp._env.action_space.labels)


##########################################################################################
# Define the Policy Network

policy_params = {'policy_class': 'deterministic',
                 'neurons': [mdp.s_dim, 50, mdp.a_dim],  # [state_dim, hidden1, ... , hiddenN, action_dim]
                 'activations': [nn.functional.relu],  # one activation function per hidden layer
                 'f_out': [lambda x: 2.0 * torch.tanh(x)]
                 }

policy = Policy(**policy_params).to(device=DEVICE, dtype=TORCH_DTYPE)

##########################################################################################
# Optimize the policy with NOPG

nopg_params = {'initial_states': np.array([env.reset() for _ in range(5)]),  # For multiple initial states
               'gamma': 0.99,
               # 'MC_samples_P': 15,
               # 'sparsify_P': {'kl_max': 0.0001, 'kl_interval_k': 20, 'kl_repeat_every_n_iterations': 200}
               }

nopg = NOPG(dataset, policy, **nopg_params)

n_policy_updates = 500
def optimizer(x): return optim.Adam(x, lr=5e-3)
evaluation_params = {'eval_mdp': mdp,
                     'eval_every_n': 100,
                     'eval_n_episodes': 1,
                     'eval_render': True
                     }

nopg.fit(n_policy_updates=n_policy_updates, optimizer=optimizer, **evaluation_params)
print()
