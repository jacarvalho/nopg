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
    id='CartpoleStabShort-v1',
    entry_point='quanser_robots.cartpole.cartpole:Cartpole',
    max_episode_steps=10000,
    kwargs={'fs': 50.0, 'fs_ctrl': 50.0, 'stabilization': True, 'long_pole': False}
)
env = gym.make('CartpoleStabShort-v1')
mdp = MDP(env)

##########################################################################################
# Gather an Off-Policy Dataset

sampling_params = {'sampling_type': 'behavioral',
                   'policy': lambda x: np.random.uniform(low=-5., high=5., size=(1,)),
                   'n_samples': 1500,
                   }

dataset = mdp.get_samples(**sampling_params)
dataset.update_dataset_internal()

# Compute Kernel Bandwidths (state, action, state next)
s_band_factor = [1., 1., 1., 1., 1.]
s_n_band_factor = s_band_factor
a_band_factor = [20.]
dataset.kde_bandwidths_internal(s_band_factor=s_band_factor, a_band_factor=a_band_factor,
                                s_n_band_factor=s_n_band_factor)

##########################################################################################
# Define the Policy Network

policy_params = {'policy_class': 'stochastic',
                 'neurons': [mdp.s_dim, 50, mdp.a_dim],  # [state_dim, hidden1, ... , hiddenN, action_dim]
                 'activations': [nn.functional.relu],  # one activation function per hidden layer
                 'f_out': [lambda x: 5.0 * torch.tanh(x), lambda x: 2.0 * torch.sigmoid(x)]
                 }

policy = Policy(**policy_params).to(device=DEVICE, dtype=TORCH_DTYPE)

##########################################################################################
# Optimize the policy with NOPG

nopg_params = {'initial_states': np.array([env.reset() for _ in range(15)]),  # For multiple initial states
               'gamma': 0.99,
               'MC_samples_stochastic_policy': 20,
               # 'sparsify_P': {'kl_max': 0.001, 'kl_interval_k': 20, 'kl_repeat_every_n_iterations': 200}
               }

nopg = NOPG(dataset, policy, **nopg_params)

n_policy_updates = 1000
def optimizer(x): return optim.Adam(x, lr=1e-2)
evaluation_params = {'eval_mdp': mdp,
                     'eval_every_n': 200,
                     'eval_n_episodes': 1,
                     'eval_render': False
                     }

nopg.fit(n_policy_updates=n_policy_updates, optimizer=optimizer, **evaluation_params)
