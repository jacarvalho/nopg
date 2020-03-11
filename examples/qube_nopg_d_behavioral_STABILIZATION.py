import math
import gym
from gym.envs import register
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from src.mdp.mdp import MDP
from src.policy.policy import Policy, policy_gmm_5volts, policy_uniform
from src.utils.utils import DEVICE, TORCH_DTYPE, NP_DTYPE
from src.nopg.nopg import NOPG


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
results_dir = '/tmp/'

sampling_params = {'sampling_type': 'behavioral',
                   'initial_state': np.array([1., 0., -1., 0., 0., 0.], dtype=NP_DTYPE),
                   'transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), math.atan2(x[3], x[2]), x[4], x[5]),
                   # 'policy': policy_gmm_5volts,
                   'policy': policy_uniform,
                   'n_samples': 2000,
                   'max_ep_transitions': 150,
                   'render': True,
                   'results_dir': results_dir
                   }

dataset = mdp.get_samples(**sampling_params)

# Compute Kernel Bandwidths (state, action, state next)
s_band_factor = [1., 1., 1., 1., 1., 1.]
s_n_band_factor = s_band_factor
a_band_factor = [1.]  # Pendulum uniform
dataset.kde_bandwidths_internal(s_band_factor=s_band_factor, a_band_factor=a_band_factor,
                                s_n_band_factor=s_n_band_factor)

dataset.plot_data_kde(state_labels=mdp._env.observation_space.labels, action_labels=mdp._env.action_space.labels)

##########################################################################################
# Define the Policy Network

policy_params = {'policy_class': 'deterministic',
                 'neurons': [mdp.s_dim, 50, mdp.a_dim],  # [state_dim, hidden1, ... , hiddenN, action_dim]
                 'activations': [nn.functional.relu],  # one activation function per hidden layer
                 'f_out': [lambda x: 5.0 * torch.tanh(x)]
                 }

policy = Policy(**policy_params).to(device=DEVICE, dtype=TORCH_DTYPE)

##########################################################################################
# Optimize the policy with NOPG

nopg_params = {'initial_states': np.array([1., 0., -1., 0., 0., 0.], dtype=NP_DTYPE).reshape(1, -1),
               'gamma': 0.99,
               # 'sparsify_P': {'kl_max': 0.001, 'kl_interval_k': 20, 'kl_repeat_every_n_iterations': 200},
               # 'MC_samples_P': 20
               }

nopg = NOPG(dataset, policy, **nopg_params)

n_policy_updates = 500
def optimizer(x): return optim.Adam(x, lr=1e-4)
evaluation_params = {'eval_mdp': mdp,
                     'eval_every_n': 100,
                     'eval_n_episodes': 1,
                     'eval_initial_state': np.array([1., 0., -1., 0., 0., 0.], dtype=NP_DTYPE),  # or None
                     'eval_transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), math.atan2(x[3], x[2]), x[4], x[5]),
                     'eval_render': True,
                     'results_dir': results_dir
                     }

nopg.fit(n_policy_updates=n_policy_updates, optimizer=optimizer, **evaluation_params)
print()
