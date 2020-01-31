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


##########################################################################################
# Create the Environment (MDP)
register(
    id='Pendulum-v1',
    entry_point='gym.envs.classic_control:PendulumEnv',
    max_episode_steps=500,
)
env = gym.make('Pendulum-v1')
mdp = MDP(env)

##########################################################################################
# Gather an Off-Policy Dataset

sampling_params = {'sampling_type': 'behavioral',
                   'transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),
                   'initial_state': np.array([1., 0., 0.], dtype=NP_DTYPE),  # theta=0., theta_dot=0,
                   'policy': policy_gmm,
                   'n_samples': 1500,
                   'max_ep_transitions': 500
                   }

dataset = mdp.get_samples(**sampling_params)
dataset.update_dataset_internal()

# Compute Kernel Bandwidths (state, action, state next)
s_band_factor = [10., 10., 1.]  # Pendulum uniform
s_n_band_factor = s_band_factor
a_band_factor = [10.]  # Pendulum uniform
dataset.kde_bandwidths_internal(s_band_factor=s_band_factor, a_band_factor=a_band_factor,
                                s_n_band_factor=s_n_band_factor)

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

nopg_params = {'initial_states': np.array([-1., 0., 0.]).reshape((-1, 3)),
               'gamma': 0.97,
               'sparsify_P': {'kl_max': 0.001, 'kl_interval_k': 20, 'kl_repeat_every_n_iterations': 200},
               'MC_samples_P': 20
               }

nopg = NOPG(dataset, policy, **nopg_params)

n_policy_updates = 1000
def optimizer(x): return optim.Adam(x, lr=1e-2)
evaluation_params = {'eval_mdp': mdp,
                     'eval_every_n': 200,
                     'eval_n_episodes': 1,
                     'eval_initial_state': np.array([-1., 0., 0.]),  # or None
                     'eval_transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),  # or None
                     'eval_render': False
                     }

nopg.fit(n_policy_updates=n_policy_updates, optimizer=optimizer, **evaluation_params)
print()
