import unittest
import math
import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from gym.envs import register

from src.mdp.mdp import MDP
from src.policy.policy import Policy, policy_gmm
from src.utils.utils import DEVICE, NP_DTYPE, TORCH_DTYPE
from src.nopg.nopg import NOPG


register(
    id='Pendulum-v1',
    entry_point='gym.envs.classic_control:PendulumEnv',
    max_episode_steps=500,
)

register(
    id='CartpoleStabShort-v1',
    entry_point='quanser_robots.cartpole.cartpole:Cartpole',
    max_episode_steps=10000,
    kwargs={'fs': 500.0, 'fs_ctrl': 500.0, 'stabilization': True, 'long_pole': False}
)


class TestNOPGpendulum(unittest.TestCase):

    def test_nopg_d_pendulum_uniform(self):
        ##########################################################################################
        # Create the Environment (MDP)
        env = gym.make('Pendulum-v1')
        mdp = MDP(env)

        ##########################################################################################
        # Gather an Off-Policy Dataset
        states = mdp.discretize_space(space='state', levels=[10, 10])  # theta, theta_dot
        actions = mdp.discretize_space(space='action', levels=[2])
        sampling_params = {'sampling_type': 'uniform',
                           'states': states,
                           'actions': actions,
                           'transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),
                           'render': False
                           }

        # sampling_params = {'sampling_type': 'behavioral',
        #                    'transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),
        #                    'initial_state': np.array([1., 0., 0.], dtype=NP_DTYPE),  # theta=0., theta_dot=0,
        #                    'policy': policy_gmm,
        #                    'n_samples': 1000,  # n_trajectories: 2
        #                    'max_ep_transitions': 500
        #                    }

        dataset = mdp.get_samples(**sampling_params)
        dataset.update_dataset_internal()

        # Compute Kernel Bandwidths (state, action, state next)
        s_band_factor = [1., 1., 1.]  # Pendulum uniform
        # s_band_factor = [10., 10., 1.]  # Pendulum behavioral
        s_n_band_factor = s_band_factor
        a_band_factor = [50.]  # Pendulum uniform
        # a_band_factor = [10.]  # Pendulum behavioral
        dataset.kde_bandwidths_internal(s_band_factor=s_band_factor, a_band_factor=a_band_factor,
                                        s_n_band_factor=s_n_band_factor)

        ##########################################################################################
        # Define the Policy Network

        policy_params = {'policy_class': 'deterministic',
                         'neurons': [mdp.s_dim, 10, mdp.a_dim],  # [state_dim, hidden1, ... , hiddenN, action_dim]
                         'activations': [nn.functional.relu],  # one activation function per hidden layer
                         'f_out': [lambda x: 2.0 * torch.tanh(x)]
                         # NOTE: for a stochastic policy f_out is a list with 2 entries (mean and diagonal covariance)
                         # 'type': 'stochastic',
                         # 'f_out': [lambda x: 2.0 * torch.tanh(x), lambda x: 2.0 * torch.sigmoid(x)]
                         }

        policy = Policy(**policy_params).to(device=DEVICE, dtype=TORCH_DTYPE)

        ##########################################################################################
        # Optimize the policy with NOPG

        nopg_params = {'initial_states': np.array([-1., 0., 0.]).reshape((-1, 3)),
                       # 'initial_states': np.array([env.reset() for _ in range(20)]),  # For multiple initial states
                       'gamma': 0.97
                       # 'MC_samples_stochastic_policy': 15
                       # 'MC_samples_P': 15
                       # 'sparsify_P': {'P_sparse_k': 10 }
                       # 'sparsify_P': {'kl_max': 0.001, 'kl_interval_k': 20, 'kl_repeat_every_n_iterations': 200}
                       }

        nopg = NOPG(dataset, policy, **nopg_params)

        n_policy_updates = 10
        def optimizer(x): return optim.Adam(x, lr=1e-2)
        evaluation_params = {'eval_mdp': mdp,
                             'eval_every_n': 200,
                             'eval_n_episodes': 1,
                             'eval_initial_state': np.array([-1., 0., 0.]),  # or None
                             'eval_transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),  # or None
                             'eval_render': False
                             }

        nopg.fit(n_policy_updates=n_policy_updates, optimizer=optimizer, **evaluation_params)

    def test_nopg_s_pendulum_uniform(self):
        ##########################################################################################
        # Create the Environment (MDP)
        env = gym.make('Pendulum-v1')
        mdp = MDP(env)

        ##########################################################################################
        # Gather an Off-Policy Dataset
        states = mdp.discretize_space(space='state', levels=[10, 10])  # theta, theta_dot
        actions = mdp.discretize_space(space='action', levels=[2])
        sampling_params = {'sampling_type': 'uniform',
                           'states': states,
                           'actions': actions,
                           'transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),
                           'render': False
                           }

        dataset = mdp.get_samples(**sampling_params)
        dataset.update_dataset_internal()

        # Compute Kernel Bandwidths (state, action, state next)
        s_band_factor = [1., 1., 1.]  # Pendulum uniform
        s_n_band_factor = s_band_factor
        a_band_factor = [50.]  # Pendulum uniform
        dataset.kde_bandwidths_internal(s_band_factor=s_band_factor, a_band_factor=a_band_factor,
                                        s_n_band_factor=s_n_band_factor)

        ##########################################################################################
        # Define the Policy Network

        policy_params = {'policy_class': 'stochastic',
                         'neurons': [mdp.s_dim, 10, mdp.a_dim],  # [state_dim, hidden1, ... , hiddenN, action_dim]
                         'activations': [nn.functional.relu],  # one activation function per hidden layer
                         'f_out': [lambda x: 2.0 * torch.tanh(x), lambda x: 2.0 * torch.sigmoid(x)]
                         }

        policy = Policy(**policy_params).to(device=DEVICE, dtype=TORCH_DTYPE)

        ##########################################################################################
        # Optimize the policy with NOPG

        nopg_params = {'initial_states': np.array([-1., 0., 0.]).reshape((-1, 3)),
                       'gamma': 0.97,
                       'MC_samples_stochastic_policy': 5
                       }

        nopg = NOPG(dataset, policy, **nopg_params)

        n_policy_updates = 10
        def optimizer(x): return optim.Adam(x, lr=1e-2)
        evaluation_params = {'eval_mdp': mdp,
                             'eval_every_n': 200,
                             'eval_n_episodes': 1,
                             'eval_initial_state': np.array([-1., 0., 0.]),  # or None
                             'eval_transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),  # or None
                             'eval_render': False
                             }

        nopg.fit(n_policy_updates=n_policy_updates, optimizer=optimizer, **evaluation_params)

    def test_nopg_d_pendulum_behavioral(self):
        ##########################################################################################
        # Create the Environment (MDP)
        env = gym.make('Pendulum-v1')
        mdp = MDP(env)

        ##########################################################################################
        # Gather an Off-Policy Dataset

        sampling_params = {'sampling_type': 'behavioral',
                           'transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),
                           'initial_state': np.array([1., 0., 0.], dtype=NP_DTYPE),  # theta=0., theta_dot=0,
                           'policy': policy_gmm,
                           'n_samples': 100,
                           'max_ep_transitions': 500
                           }

        dataset = mdp.get_samples(**sampling_params)
        dataset.update_dataset_internal()

        # Compute Kernel Bandwidths (state, action, state next)
        s_band_factor = [10., 10., 1.]  # Pendulum behavioral
        s_n_band_factor = s_band_factor
        a_band_factor = [10.]  # Pendulum behavioral
        dataset.kde_bandwidths_internal(s_band_factor=s_band_factor, a_band_factor=a_band_factor,
                                        s_n_band_factor=s_n_band_factor)

        ##########################################################################################
        # Define the Policy Network

        policy_params = {'policy_class': 'deterministic',
                         'neurons': [mdp.s_dim, 10, mdp.a_dim],  # [state_dim, hidden1, ... , hiddenN, action_dim]
                         'activations': [nn.functional.relu],  # one activation function per hidden layer
                         'f_out': [lambda x: 2.0 * torch.tanh(x)]
                         }

        policy = Policy(**policy_params).to(device=DEVICE, dtype=TORCH_DTYPE)

        ##########################################################################################
        # Optimize the policy with NOPG

        nopg_params = {'initial_states': np.array([-1., 0., 0.]).reshape((-1, 3)),
                       'gamma': 0.97
                       }

        nopg = NOPG(dataset, policy, **nopg_params)

        n_policy_updates = 10

        def optimizer(x): return optim.Adam(x, lr=1e-2)

        evaluation_params = {'eval_mdp': mdp,
                             'eval_every_n': 200,
                             'eval_n_episodes': 1,
                             'eval_initial_state': np.array([-1., 0., 0.]),  # or None
                             'eval_transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),  # or None
                             'eval_render': False
                             }

        nopg.fit(n_policy_updates=n_policy_updates, optimizer=optimizer, **evaluation_params)

    def test_nopg_s_pendulum_behavioral(self):
        ##########################################################################################
        # Create the Environment (MDP)
        env = gym.make('Pendulum-v1')
        mdp = MDP(env)

        ##########################################################################################
        # Gather an Off-Policy Dataset

        sampling_params = {'sampling_type': 'behavioral',
                           'transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),
                           'initial_state': np.array([1., 0., 0.], dtype=NP_DTYPE),  # theta=0., theta_dot=0,
                           'policy': policy_gmm,
                           'n_samples': 100,  # n_trajectories: 2
                           'max_ep_transitions': 500
                           }

        dataset = mdp.get_samples(**sampling_params)
        dataset.update_dataset_internal()

        # Compute Kernel Bandwidths (state, action, state next)
        s_band_factor = [10., 10., 1.]  # Pendulum behavioral
        s_n_band_factor = s_band_factor
        a_band_factor = [10.]  # Pendulum behavioral
        dataset.kde_bandwidths_internal(s_band_factor=s_band_factor, a_band_factor=a_band_factor,
                                        s_n_band_factor=s_n_band_factor)

        ##########################################################################################
        # Define the Policy Network

        policy_params = {'policy_class': 'stochastic',
                         'neurons': [mdp.s_dim, 10, mdp.a_dim],  # [state_dim, hidden1, ... , hiddenN, action_dim]
                         'activations': [nn.functional.relu],  # one activation function per hidden layer
                         'f_out': [lambda x: 2.0 * torch.tanh(x), lambda x: 2.0 * torch.sigmoid(x)]
                         }

        policy = Policy(**policy_params).to(device=DEVICE, dtype=TORCH_DTYPE)

        ##########################################################################################
        # Optimize the policy with NOPG

        nopg_params = {'initial_states': np.array([-1., 0., 0.]).reshape((-1, 3)),
                       'gamma': 0.97,
                       'MC_samples_stochastic_policy': 15
                       }

        nopg = NOPG(dataset, policy, **nopg_params)

        n_policy_updates = 10

        def optimizer(x): return optim.Adam(x, lr=1e-2)

        evaluation_params = {'eval_mdp': mdp,
                             'eval_every_n': 200,
                             'eval_n_episodes': 1,
                             'eval_initial_state': np.array([-1., 0., 0.]),  # or None
                             'eval_transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),  # or None
                             'eval_render': False
                             }

        nopg.fit(n_policy_updates=n_policy_updates, optimizer=optimizer, **evaluation_params)

    def test_nopg_d_pendulum_uniform_initial_states(self):
        ##########################################################################################
        # Create the Environment (MDP)
        env = gym.make('Pendulum-v1')
        mdp = MDP(env)

        ##########################################################################################
        # Gather an Off-Policy Dataset
        states = mdp.discretize_space(space='state', levels=[10, 10])  # theta, theta_dot
        actions = mdp.discretize_space(space='action', levels=[2])
        sampling_params = {'sampling_type': 'uniform',
                           'states': states,
                           'actions': actions,
                           'transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),
                           'render': False
                           }

        dataset = mdp.get_samples(**sampling_params)
        dataset.update_dataset_internal()

        # Compute Kernel Bandwidths (state, action, state next)
        s_band_factor = [1., 1., 1.]  # Pendulum uniform
        s_n_band_factor = s_band_factor
        a_band_factor = [50.]  # Pendulum uniform
        dataset.kde_bandwidths_internal(s_band_factor=s_band_factor, a_band_factor=a_band_factor,
                                        s_n_band_factor=s_n_band_factor)

        ##########################################################################################
        # Define the Policy Network

        policy_params = {'policy_class': 'deterministic',
                         'neurons': [mdp.s_dim, 10, mdp.a_dim],  # [state_dim, hidden1, ... , hiddenN, action_dim]
                         'activations': [nn.functional.relu],  # one activation function per hidden layer
                         'f_out': [lambda x: 2.0 * torch.tanh(x)]
                         }

        policy = Policy(**policy_params).to(device=DEVICE, dtype=TORCH_DTYPE)

        ##########################################################################################
        # Optimize the policy with NOPG

        nopg_params = {'initial_states': np.array([env.reset() for _ in range(20)]),  # For multiple initial states
                       'gamma': 0.97
                       }

        nopg = NOPG(dataset, policy, **nopg_params)

        n_policy_updates = 10
        def optimizer(x): return optim.Adam(x, lr=1e-2)
        evaluation_params = {'eval_mdp': mdp,
                             'eval_every_n': 200,
                             'eval_n_episodes': 1,
                             'eval_initial_state': np.array([-1., 0., 0.]),  # or None
                             'eval_transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),  # or None
                             'eval_render': False
                             }

        nopg.fit(n_policy_updates=n_policy_updates, optimizer=optimizer, **evaluation_params)

    def test_nopg_s_pendulum_behavioral_sample_P(self):
        ##########################################################################################
        # Create the Environment (MDP)
        env = gym.make('Pendulum-v1')
        mdp = MDP(env)

        ##########################################################################################
        # Gather an Off-Policy Dataset
        sampling_params = {'sampling_type': 'behavioral',
                           'transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),
                           'initial_state': np.array([1., 0., 0.], dtype=NP_DTYPE),  # theta=0., theta_dot=0,
                           'policy': policy_gmm,
                           'n_samples': 100,  # n_trajectories: 2
                           'max_ep_transitions': 500
                           }

        dataset = mdp.get_samples(**sampling_params)
        dataset.update_dataset_internal()

        # Compute Kernel Bandwidths (state, action, state next)
        s_band_factor = [10., 10., 1.]  # Pendulum behavioral
        s_n_band_factor = s_band_factor
        a_band_factor = [10.]  # Pendulum behavioral
        dataset.kde_bandwidths_internal(s_band_factor=s_band_factor, a_band_factor=a_band_factor,
                                        s_n_band_factor=s_n_band_factor)

        ##########################################################################################
        # Define the Policy Network

        policy_params = {'policy_class': 'stochastic',
                         'neurons': [mdp.s_dim, 10, mdp.a_dim],  # [state_dim, hidden1, ... , hiddenN, action_dim]
                         'activations': [nn.functional.relu],  # one activation function per hidden layer
                         'f_out': [lambda x: 2.0 * torch.tanh(x), lambda x: 2.0 * torch.sigmoid(x)]
                         }

        policy = Policy(**policy_params).to(device=DEVICE, dtype=TORCH_DTYPE)

        ##########################################################################################
        # Optimize the policy with NOPG

        nopg_params = {'initial_states': np.array([-1., 0., 0.]).reshape((-1, 3)),
                       'gamma': 0.97,
                       'MC_samples_P': 2,
                       'MC_samples_stochastic_policy': 2
                       }

        nopg = NOPG(dataset, policy, **nopg_params)

        n_policy_updates = 10
        def optimizer(x): return optim.Adam(x, lr=1e-2)
        evaluation_params = {'eval_mdp': mdp,
                             'eval_every_n': 200,
                             'eval_n_episodes': 1,
                             'eval_initial_state': np.array([-1., 0., 0.]),  # or None
                             'eval_transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),  # or None
                             'eval_render': False
                             }

        nopg.fit(n_policy_updates=n_policy_updates, optimizer=optimizer, **evaluation_params)

    def test_nopg_d_pendulum_uniform_sparsify_P(self):
        ##########################################################################################
        # Create the Environment (MDP)
        env = gym.make('Pendulum-v1')
        mdp = MDP(env)

        ##########################################################################################
        # Gather an Off-Policy Dataset
        states = mdp.discretize_space(space='state', levels=[10, 10])  # theta, theta_dot
        actions = mdp.discretize_space(space='action', levels=[2])
        sampling_params = {'sampling_type': 'uniform',
                           'states': states,
                           'actions': actions,
                           'transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),
                           'render': False
                           }

        dataset = mdp.get_samples(**sampling_params)
        dataset.update_dataset_internal()

        # Compute Kernel Bandwidths (state, action, state next)
        s_band_factor = [1., 1., 1.]  # Pendulum uniform
        s_n_band_factor = s_band_factor
        a_band_factor = [50.]  # Pendulum uniform
        dataset.kde_bandwidths_internal(s_band_factor=s_band_factor, a_band_factor=a_band_factor,
                                        s_n_band_factor=s_n_band_factor)

        ##########################################################################################
        # Define the Policy Network

        policy_params = {'policy_class': 'deterministic',
                         'neurons': [mdp.s_dim, 10, mdp.a_dim],  # [state_dim, hidden1, ... , hiddenN, action_dim]
                         'activations': [nn.functional.relu],  # one activation function per hidden layer
                         'f_out': [lambda x: 2.0 * torch.tanh(x)]
                         }

        policy = Policy(**policy_params).to(device=DEVICE, dtype=TORCH_DTYPE)

        ##########################################################################################
        # Optimize the policy with NOPG

        nopg_params = {'initial_states': np.array([-1., 0., 0.]).reshape((-1, 3)),
                       'gamma': 0.97,
                       'sparsify_P': {'P_sparse_k': 10}
                       }

        nopg = NOPG(dataset, policy, **nopg_params)

        n_policy_updates = 10
        def optimizer(x): return optim.Adam(x, lr=1e-2)
        evaluation_params = {'eval_mdp': mdp,
                             'eval_every_n': 200,
                             'eval_n_episodes': 1,
                             'eval_initial_state': np.array([-1., 0., 0.]),  # or None
                             'eval_transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),  # or None
                             'eval_render': False
                             }

        nopg.fit(n_policy_updates=n_policy_updates, optimizer=optimizer, **evaluation_params)

    def test_nopg_d_pendulum_uniform_sparsify_P_kl_divergence(self):
        ##########################################################################################
        # Create the Environment (MDP)
        env = gym.make('Pendulum-v1')
        mdp = MDP(env)

        ##########################################################################################
        # Gather an Off-Policy Dataset
        states = mdp.discretize_space(space='state', levels=[10, 10])  # theta, theta_dot
        actions = mdp.discretize_space(space='action', levels=[2])
        sampling_params = {'sampling_type': 'uniform',
                           'states': states,
                           'actions': actions,
                           'transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),
                           'render': False
                           }

        dataset = mdp.get_samples(**sampling_params)
        dataset.update_dataset_internal()

        # Compute Kernel Bandwidths (state, action, state next)
        s_band_factor = [1., 1., 1.]  # Pendulum uniform
        s_n_band_factor = s_band_factor
        a_band_factor = [50.]  # Pendulum uniform
        dataset.kde_bandwidths_internal(s_band_factor=s_band_factor, a_band_factor=a_band_factor,
                                        s_n_band_factor=s_n_band_factor)

        ##########################################################################################
        # Define the Policy Network

        policy_params = {'policy_class': 'deterministic',
                         'neurons': [mdp.s_dim, 10, mdp.a_dim],  # [state_dim, hidden1, ... , hiddenN, action_dim]
                         'activations': [nn.functional.relu],  # one activation function per hidden layer
                         'f_out': [lambda x: 2.0 * torch.tanh(x)]
                         }

        policy = Policy(**policy_params).to(device=DEVICE, dtype=TORCH_DTYPE)

        ##########################################################################################
        # Optimize the policy with NOPG

        nopg_params = {'initial_states': np.array([-1., 0., 0.]).reshape((-1, 3)),
                       'gamma': 0.97,
                       'sparsify_P': {'kl_max': 0.001, 'kl_interval_k': 10, 'kl_repeat_every_n_iterations': 3}
                       }

        nopg = NOPG(dataset, policy, **nopg_params)

        n_policy_updates = 10
        def optimizer(x): return optim.Adam(x, lr=1e-2)
        evaluation_params = {'eval_mdp': mdp,
                             'eval_every_n': 200,
                             'eval_n_episodes': 1,
                             'eval_initial_state': np.array([-1., 0., 0.]),  # or None
                             'eval_transform_to_internal_state': lambda x: (math.atan2(x[1], x[0]), x[2]),  # or None
                             'eval_render': False
                             }

        nopg.fit(n_policy_updates=n_policy_updates, optimizer=optimizer, **evaluation_params)


class TestNOPGcartpole(unittest.TestCase):

    def test_nopg_d_cartpole_behavioral(self):
        ##########################################################################################
        # Create the Environment (MDP)
        env = gym.make('CartpoleStabShort-v1')
        mdp = MDP(env)

        ##########################################################################################
        # Gather an Off-Policy Dataset

        sampling_params = {'sampling_type': 'behavioral',
                           'policy': lambda x: np.random.uniform(low=-5., high=5., size=(1,)),
                           'n_samples': 200,  # n_trajectories: 2
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

        policy_params = {'policy_class': 'deterministic',
                         'neurons': [mdp.s_dim, 10, mdp.a_dim],  # [state_dim, hidden1, ... , hiddenN, action_dim]
                         'activations': [nn.functional.relu],  # one activation function per hidden layer
                         'f_out': [lambda x: 5.0 * torch.tanh(x)]
                         }

        policy = Policy(**policy_params).to(device=DEVICE, dtype=TORCH_DTYPE)

        ##########################################################################################
        # Optimize the policy with NOPG

        nopg_params = {'initial_states': np.array([env.reset() for _ in range(5)]),  # For multiple initial states
                       'gamma': 0.99
                       }

        nopg = NOPG(dataset, policy, **nopg_params)

        n_policy_updates = 10
        def optimizer(x): return optim.Adam(x, lr=1e-2)
        evaluation_params = {'eval_mdp': mdp,
                             'eval_every_n': 200,
                             'eval_n_episodes': 1,
                             'eval_render': False
                             }

        nopg.fit(n_policy_updates=n_policy_updates, optimizer=optimizer, **evaluation_params)


if __name__ == '__main__':
    unittest.main()
