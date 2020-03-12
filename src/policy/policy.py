import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal
import getch

from src.configs.configs import TORCH_DTYPE


class Policy(nn.Module):

    def __init__(self, policy_class='deterministic', neurons=None, activations=None, f_out=None, device=None):
        """
        Policy parametrized by a neural network.
        A deterministic policy outputs a single action. A stochastic policy outputs the parameters of a Gaussian, the
        mean and the diagonal covariance.

        :param policy_class: 'deterministic' or 'stochastic'
        :type policy_class: str
        :param neurons: list of hidden neurons per layer, including state and action dimensions
            [state_dim, h1, ..., hN, action_dim]
        :type neurons: list (int)
        :param activations: list of hidden neuron activations
        :type activations: list (torch.nn.functional)
        :param f_out: function to apply to the output of the neural net. If using a deterministic policy, it has just
            one entry, with a stochastic policy use two entries (for mean and covariance)
        :type f_out: list (function)
        """
        super(Policy, self).__init__()

        self.device = device

        # Policy paameters
        self.policy_class = policy_class

        # Neural Net Parameters
        self._input_dim = neurons[0]
        self._output_dim = neurons[-1] if policy_class == 'deterministic' else 2*neurons[-1]
        self._fc = nn.ModuleList()
        for i in range(1, len(neurons)-1):
            self._fc.append(nn.Linear(neurons[i-1], neurons[i]))
            nn.init.xavier_uniform_(self._fc[-1].weight)
        self._fc.append(nn.Linear(neurons[-2], neurons[-1] if policy_class == 'deterministic' else 2*neurons[-1]))
        nn.init.xavier_uniform_(self._fc[-1].weight)

        # Neuron activations
        self._activations = []
        for activation in activations:
            self._activations.append(activation)

        # Output functions
        self._f_out = f_out

    def forward(self, state):
        """
        Compute the action for a given state.
        In case of a stochastic policy this includes sampling from the resulting Gaussian distribution.

        :param state: a tensor, dimensions (batch_size, state_dim)
        :type state: torch.tensor
        :return: an action a=pi(state), or a~pi(.|state), dimensions (batch_size, state_dim)
        """
        h = state
        for i in range(len(self._fc)-1):
            h = self._activations[i](self._fc[i](h))
        out = self._fc[-1](h)

        if self.policy_class == 'stochastic':
            # A stochastic policy returns the mean and diagonal covariance of a Gaussian
            if out.dim() > 1:
                out = out[:, 0:self._output_dim//2], out[:, self._output_dim//2:]
            else:
                out = out[0:self._output_dim//2], out[self._output_dim//2:]
            mean, cov = self._f_out[0](out[0]), self._f_out[1](out[1])
            # Sample an action from the Gaussian stochastic policy with the reparametrization trick
            n_dim = mean.shape[1] if mean.dim() > 1 else mean.shape[0]
            m = MultivariateNormal(loc=torch.zeros(n_dim, device=self.device, dtype=TORCH_DTYPE),
                                   covariance_matrix=torch.eye(n_dim, device=self.device, dtype=TORCH_DTYPE))
            out = mean + cov * m.sample(sample_shape=(mean.shape[0],))
        elif self.policy_class == 'deterministic':
            out = self._f_out[0](out)
        return out


def policy_gmm(x=None):
    """
    Samples from a Gaussian Mixture Model.
    """
    means = [-2., 2.]
    z = np.random.choice(2)
    return np.random.normal(loc=means[z], scale=1.0, size=(1,))


def policy_gmm_5volts(x=None):
    """
    Samples from a Gaussian Mixture Model.
    """
    means = [-5., 5.]
    z = np.random.choice(2)
    return np.random.normal(loc=means[z], scale=3.0, size=(1,))


def policy_uniform(x=None, a=5.):
    """
    Samples from a Uniform distribution.
    """
    return np.random.uniform(low=-a, high=a, size=(1,))


def mountaincar_policy(x=None):
    """
    Actions from the user for the mountaincar environment.
    """
    noise = 0.001
    c = getch.getch()
    if c == 'a':
        a = np.random.normal(loc=-1, scale=noise, size=(1,))
    elif c == 'd':
        a = np.random.normal(loc=+1, scale=noise, size=(1,))
    else:
        a = np.zeros((1,))
    return a

