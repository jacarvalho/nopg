import numpy as np
import math
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import os
import matplotlib.pyplot as plt

from src.configs.configs import NP_DTYPE

# Matplotlib settings
plt.style.use('seaborn-darkgrid')
plt.rcParams['savefig.format'] = 'png'


class Dataset:

    def __init__(self, results_dir='/tmp/'):
        """
        Class to manage transitions (state, action, reward, state_next, dones).
        """
        self._trajectories = []
        self._states = None
        self._actions = None
        self._rewards = None
        self._states_next = None
        self._dones = None
        self._s_bandwidth = None
        self._a_bandwidth = None
        self._s_n_bandwidth = None
        self._results_dir = os.path.abspath(results_dir)

    def add_trajectory(self, trajectory):
        """
        Appends a trajectory. A trajectory is a list of tuples with transitions (s, a, r, s_n, d).

        :param trajectory: list of transitions
        :type trajectory: list
        """
        self._trajectories.append(trajectory)

    def update_dataset_internal(self, n_trajectories=-1):
        """
        Concatenates all states, actions, rewards, states_next and dones stored as trajectories into their internal
        arrays.

        :param n_trajectories: number of trajectories to update (-1 is default for all trajectories)
        :type: int
        """
        states, actions, rewards, states_next, dones = [], [], [], [], []
        if n_trajectories == -1:
            for trajectory in self._trajectories:
                for transition in trajectory:
                    s, a, r, s_n, d = transition
                    states.append(s)
                    actions.append(a)
                    rewards.append(r)
                    states_next.append(s_n)
                    dones.append(d)
        else:
            raise NotImplementedError
        self._states = np.array(states, dtype=NP_DTYPE)
        self._actions = np.array(actions, dtype=NP_DTYPE)
        self._rewards = np.array(rewards, dtype=NP_DTYPE).reshape((-1, 1))
        self._states_next = np.array(states_next, dtype=NP_DTYPE)
        self._dones = np.array(dones, dtype=NP_DTYPE).reshape((-1, 1))

    def get_full_batch(self):
        self.update_dataset_internal()
        return self._states, self._actions, self._rewards, self._states_next, self._dones

    def get_bandwidths(self):
        return self._s_bandwidth, self._a_bandwidth, self._s_n_bandwidth

    def kde_bandwidths_internal(self, s_band_factor=None, a_band_factor=None,
                                s_n_band_factor=None, n_bandwidths=30, cv_folds=5):
        """
        Compute KDE bandwidths for each dimension of the state and action spaces and update the class internal members,
        by using Cross-Validation.

        :param s_band_factor: state bandwidth factors
        :type s_band_factor: list (int)
        :param a_band_factor: action bandwidth factors
        :type a_band_factor: list (int)
        :param s_n_band_factor: state_next bandwidth factors
        :type s_n_band_factor: list (int)
        :param n_bandwidths: number of bandwidths for cross-validation
        :type n_bandwidths: int
        :param cv_folds: k-fold cross validation
        :type cv_folds: int
        """
        self.update_dataset_internal()
        states, actions, states_next = self._states, self._actions, self._states_next
        print("Computing kernel bandwidths with Cross-Validation...")
        print('#samples: {}'.format(states.shape[0]))
        datasets = [states, actions, states_next]
        band_factors = [s_band_factor, a_band_factor, s_n_band_factor]
        for i, data in enumerate(datasets):
            bandwidths = [1.]*data.shape[1]
            for j in range(data.shape[1]):
                print("\rdataset {}/{}, dim {}/{}".format(i+1, len(datasets), j+1, data.shape[1]), end=' ', flush=True)
                x = data[:, j].reshape((-1, 1))
                # Compute an initial estimate for the bandwith. Silverman bandwith expects a Gaussian distribution.
                kde_gaussian = gaussian_kde(x.reshape(-1), bw_method='silverman')
                std_init = math.sqrt(kde_gaussian.covariance.item())
                # Compute the bandwidth with Cross Validation.
                inf_std, sup_std = std_init * 0.0, std_init * 1.5
                kde_grid = GridSearchCV(
                    KernelDensity(kernel='gaussian'),
                    {'bandwidth': np.linspace(inf_std, sup_std, num=n_bandwidths)}, cv=cv_folds, n_jobs=-1)
                kde_grid.fit(x)
                h_best = kde_grid.best_params_['bandwidth']

                try:
                    bandwidths[j] = h_best * band_factors[i][j]
                except IndexError:
                    bandwidths[j] = h_best * 1.0
                except TypeError:
                    bandwidths[j] = h_best * 1.0

            if i == 0:
                self._s_bandwidth = np.array(bandwidths)
            elif i == 1:
                self._a_bandwidth = np.array(bandwidths)
            elif i == 2:
                self._s_n_bandwidth = np.array(bandwidths)

    def save_trajectories_to_file(self, filename=None):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            arr = np.array([np.array(xi) for xi in self._trajectories])
            np.save(filename, arr)

    def load_trajectories_from_file(self, filename=None, n_trajectories=None):
        """
        Loads trajectories from filename to the internal member.
        If n_trajectories is equal to -1 returns all trajectories, else samples uniformly n_trajectories from the
        loaded trajectories.

        :param filename:
        :param n_trajectories:
        :return:
        """
        trajectories = np.load(filename, allow_pickle=True)
        if n_trajectories == -1:
            self._trajectories = trajectories
        else:
            idxs = np.random.choice(np.arange(trajectories.shape[0]), replace=False,
                                    size=min(n_trajectories, len(trajectories)))
            self._trajectories = trajectories[idxs]

    def plot_data_kde(self, state_labels=None, action_labels=None):
        self.update_dataset_internal()
        n_granularity = 1000
        # States
        fig, axs = plt.subplots(nrows=self._states.shape[1], ncols=1, figsize=(20, 10))
        if not isinstance(axs, np.ndarray):
            axs = np.array(axs).reshape((1,))
        for dim in range(self._states.shape[1]):
            xs = np.linspace(np.min(self._states[:, dim]), np.max(self._states[:, dim]), n_granularity)
            kde = gaussian_kde(self._states[:, dim], bw_method=1)
            kde.set_bandwidth(bw_method=self._s_bandwidth[dim]**2)
            axs[dim].plot(xs, kde.pdf(xs))
            axs[dim].set_xlabel(state_labels[dim] if state_labels else "s_{}".format(dim))
        plt.tight_layout()
        fig_path = os.path.join(self._results_dir, 'kde_states')
        fig.savefig(fig_path)

        # Actions
        fig, axs = plt.subplots(nrows=self._actions.shape[1], ncols=1, figsize=(20, 10))
        if not isinstance(axs, np.ndarray):
            axs = np.array(axs).reshape((1,))
        for dim in range(self._actions.shape[1]):
            xs = np.linspace(np.min(self._actions[:, dim]), np.max(self._actions[:, dim]), n_granularity)
            kde = gaussian_kde(self._actions[:, dim], bw_method=1)
            kde.set_bandwidth(bw_method=self._a_bandwidth[dim]**2)
            axs[dim].plot(xs, kde.pdf(xs))
            axs[dim].set_xlabel(action_labels[dim] if action_labels else "a_{}".format(dim))
        plt.tight_layout()
        fig_path = os.path.join(self._results_dir, 'kde_actions')
        fig.savefig(fig_path)
