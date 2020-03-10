import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import bicgstab

from src.utils.utils import DEVICE, TORCH_DTYPE, NP_DTYPE


class NOPG:

    def __init__(self, dataset, policy, initial_states=None, gamma=0.99, MC_samples_stochastic_policy=None,
                 MC_samples_P=None, sparsify_P=None):
        """
        Implements the NOPG algorithm. The minimum to be provided are the dataset, the policy to be optimized, and an
        initial state.

        :param dataset: a dataset containing pre-collected trajectories
        :type dataset: Dataset
        :param policy: the parametrized policy to be optimized
        :type policy: Policy
        :param initial_states: one initial states or a set of initial states (n_initial_states, state_dim)
        :type initial_states: ndarray
        :param gamma: discount factor  (0, 1)
        :type gamma: float
        :param MC_samples_stochastic_policy: number of monte carlo samples for the stochastic policy
        :type MC_samples_stochastic_policy: int
        :param MC_samples_P: number of monte carlo samples for the P matrix computation. If None, the mean is used
        :type MC_samples_P: int
        :param sparsify_P: dictionary with parameters for the sparsification of the P matrix, to construct
            Lambda=I-gamma*P. There are two types of sparsification, fixed largest k or kl divergence based
            {'P_sparse_k': (int) keep always the largest k values per row}
            {'kl_max': (float) maximum average kl divergence per row,
            'kl_interval_k': (int) evaluate the largest k values per row with this interval,
            'kl_repeat_every_n_iterations': (int) recompute kl divergence based sparsification every
                n iterations of the policy gradient }
        :type sparsify_P: dict
        """
        # Dataset
        self._dataset = dataset
        self._states, self._actions, self._rewards, self._states_next, self._dones, \
            self._s_bandwidth, self._a_bandwidth, self._s_n_bandwidth = self.prepare_data()
        self._initial_states = torch.tensor(initial_states, device=DEVICE, dtype=TORCH_DTYPE)
        self._n_samples = self._states.shape[0]

        # Policy
        self._policy = policy
        if self._policy.policy_class == 'stochastic':
            self._MC_samples_stochastic_policy = MC_samples_stochastic_policy

        # NOPG internals
        self._gamma = gamma
        self._eps = None
        self._eps_0 = None
        self._P = None
        self._MC_samples_P = MC_samples_P
        self._sparsify_P = False
        if sparsify_P is not None:
            self._sparsify_P = True
            self._P_sparse = None
            if 'P_sparse_k' in sparsify_P:
                self._sparse_P_option = 'largest_k'
                self._P_sparse_k = sparsify_P['P_sparse_k']
            else:
                self._sparse_P_option = 'kl_div'
                self._P_sparse_kl_max = sparsify_P['kl_max']
                self._P_sparse_kl_interval_k = sparsify_P['kl_interval_k']
                self._P_sparse_kl_repeat_every_n_iterations = sparsify_P['kl_repeat_every_n_iterations']
        self._Lambda = None
        self._Lambda_sparse = None
        self._q = None
        self._mu = None
        self._J = None
        self._objective = None
        self._iter = 0

    def prepare_data(self):
        """
        Convert dataset from numpy to torch tensors.

        :return: states, actions, rewards, states_next, dones, state_bandwidths, action_bandwidths,
            state_next_bandwidths
        """
        states, actions, rewards, states_next, dones = self._dataset.get_full_batch()
        s_bandwidth, a_bandwidth, s_n_bandwidth = self._dataset.get_bandwidths()
        return torch.tensor(states, device=DEVICE, dtype=TORCH_DTYPE),\
            torch.tensor(actions, device=DEVICE, dtype=TORCH_DTYPE),\
            torch.tensor(rewards, device=DEVICE, dtype=TORCH_DTYPE),\
            torch.tensor(states_next, device=DEVICE, dtype=TORCH_DTYPE),\
            torch.tensor(dones, device=DEVICE, dtype=TORCH_DTYPE),\
            torch.tensor(s_bandwidth, device=DEVICE, dtype=TORCH_DTYPE),\
            torch.tensor(a_bandwidth, device=DEVICE, dtype=TORCH_DTYPE),\
            torch.tensor(s_n_bandwidth, device=DEVICE, dtype=TORCH_DTYPE)

    def build_full_P(self):
        """
        Builds the FULL P matrix.
        """
        if self._MC_samples_P is not None:
            kernel_s_n = MultivariateNormal(loc=self._states_next.repeat(self._MC_samples_P, 1),
                                            covariance_matrix=torch.diag(self._s_n_bandwidth.pow(2)))
            self._P = self.eps_matrix(kernel_s_n.sample(), self._states).reshape(
                self._MC_samples_P, self._n_samples, self._n_samples).mean(dim=0)

        else:
            kernel_s_n = MultivariateNormal(loc=self._states_next,
                                            covariance_matrix=torch.diag(self._s_n_bandwidth.pow(2)))
            self._P = self.eps_matrix(kernel_s_n.mean, self._states)
        # In terminal states, the rows of P are set to zero
        idxs_terminal = self._dones.nonzero(as_tuple=True)[0]
        self._P[idxs_terminal, :] = 0.

    def build_P_sparse_largest_k(self, k):
        """
        Builds a sparse P matrix by selecting the largest k elements of each row and zeroing out the rest.

        :param k: number of entries to select per row
        :type k: int
        :return: a sparse tensor with the sparse P matrix
        """
        # In terminal states, the rows of P are set to zero
        idxs_non_terminal = (self._dones == 0).nonzero(as_tuple=True)[0]
        # Get the largest k values for each row of P
        vals_idx = torch.sort(self._P[idxs_non_terminal, :], dim=1, descending=True)
        vals, idx = vals_idx[0][:, 0:k], vals_idx[1][:, 0:k]
        # Sparse matrix indexes
        idxs_rows = idxs_non_terminal.reshape((-1, 1)).repeat(1, k).reshape((-1,))
        idxs_cols = idx.reshape((-1,))
        # Normalize the values along rows, such that P_sparse remains a stochastic matrix
        vals = self.normalize(vals, dim=1).reshape((-1,))
        return torch.sparse_coo_tensor(torch.stack((idxs_rows, idxs_cols), dim=0), vals,
                                       size=(self._n_samples, self._n_samples), device=DEVICE, dtype=TORCH_DTYPE)

    def build_P_sparse(self):
        """
        Builds a sparse P matrix either by selecting the largest elements of each row, using a fixed number, or by
        adapting the number of elements to keep per row, such that a maximum value for the average KL divergence is
        attained.
        """
        if self._sparse_P_option == 'largest_k':
            self._P_sparse = self.build_P_sparse_largest_k(self._P_sparse_k)
        elif self._sparse_P_option == 'kl_div':
            if self._iter % self._P_sparse_kl_repeat_every_n_iterations == 0:
                print("\nComputing KL(P|Psparse)", end=" ")
                avg_kl_div = math.inf
                i = 1
                while avg_kl_div > self._P_sparse_kl_max and i * self._P_sparse_kl_interval_k <= self._n_samples:
                    P_sparse_dense = self.build_P_sparse_largest_k(i * self._P_sparse_kl_interval_k).to_dense()
                    avg_kl_div = F.kl_div(torch.log(self._P), P_sparse_dense, reduction='batchmean').item()
                    self._P_sparse_k = i * self._P_sparse_kl_interval_k
                    i += 1
                if i >= self._n_samples:
                    self._P_sparse_k = self._n_samples
                print("| k = {}".format(self._P_sparse_k))
            self._P_sparse = self.build_P_sparse_largest_k(self._P_sparse_k)
        else:
            raise NotImplementedError

    def build_Lambda(self):
        """
        Builds the Lambda matrix, L = I - gamma*P. Lambda can be sparse if P is also sparse.
        """
        with torch.no_grad():
            if self._sparsify_P:
                self.build_P_sparse()
                self._eye_sparse = torch.sparse_coo_tensor(
                    torch.arange(self._n_samples).repeat(2, 1), torch.ones(self._n_samples),
                    size=(self._n_samples, self._n_samples), device=DEVICE, dtype=TORCH_DTYPE)
                self._Lambda_sparse = self._eye_sparse.add(-self._gamma * self._P_sparse)
            else:
                self._Lambda = torch.eye(self._n_samples, device=DEVICE, dtype=TORCH_DTYPE) - self._gamma * self._P

    def build_eps_0(self):
        """
        Builds the vector eps0 (n_initial_states, n_samples).
        """
        self._eps_0 = self.eps_matrix(self._initial_states, self._states).t()

    def eps_matrix(self, X, Y):
        """
        Builds the epsilon vector/matrix between X and Y, where X are the input points and Y are the support points.
        If X is one query point, the result is a vector. If X is a batch of points, the result is a matrix.

        :param X: the input states to build the kernel matrix (batchsize, state_dim)
        :type X: torch.tensor
        :param Y: the state support points (batchsize, state_dim)
        :type Y: torch.tensor
        :return: stochastic epsilon matrix between X and Y (batchsize of X, batchsize of Y)
        """
        s_kernel = self.kernel(X, Y, self._s_bandwidth)
        a_kernel = None
        if self._policy.policy_class == 'stochastic':
            a_kernel = self.kernel(self._policy(X.repeat(self._MC_samples_stochastic_policy, 1)), self._actions,
                                   self._a_bandwidth)
            a_kernel = a_kernel.reshape(self._MC_samples_stochastic_policy, X.shape[0], Y.shape[0]).mean(dim=0)
        elif self._policy.policy_class == 'deterministic':
            a_kernel = self.kernel(self._policy(X), self._actions, self._a_bandwidth)
        return self.normalize(s_kernel * a_kernel, dim=1)

    @staticmethod
    def kernel(X, Y, bandwidth):
        """
        Computes the Kernel matrix between X and Y, with a multivariate Gaussian kernel.
        Bandwidth is an array with the diagonal of the Scale matrix (Scale.T @ Scale = Covariance).
        NOTE: This is NOT the variance of each dimension, but the standard deviation.
        Given Y is (N,d), N points, d-dimensional, if X is one data point, it returns a row vector with
        dimensions (1,N). If x is an array with P points, it returns a matrix with dimensions (P,N), where each row
        is a vector with the kernel transformations of x_1,...,x_P, with N being the number of kernel support points.

        :param X: input data
        :type X: torch.tensor
        :param Y: kernel support points
        :type Y: torch.tensor
        :param bandwidth: array with diagonal bandwidth
        :type bandwidth: torch.tensor
        :return: a matrix with the kernel between X and points
        """
        n_points = Y.shape[0]
        kernels = MultivariateNormal(loc=Y, covariance_matrix=torch.diag(bandwidth.pow(2)))
        X = (X.unsqueeze(dim=1)).repeat(1, n_points, 1)
        K = kernels.log_prob(X)
        return torch.exp(K)

    @staticmethod
    def normalize(X, dim=1):
        """
        Normalize X to sum up to 1 along dim, with l1-norm.

        :param X: the tensor to normalize
        :type X: torch.tensor
        :param dim: dimension
        :type dim: int
        :return: a normalized tensor, summing to 1 along dim
        """
        return F.normalize(X, p=1, dim=dim)

    def compute_q_mu(self):
        """
        Computes the q and mu vectors using the Conjugate Gradient method for sparse linear systems.
        """
        if self._sparsify_P:
            idxs = self._Lambda_sparse.to('cpu')._indices().numpy()
            vals = self._Lambda_sparse.to('cpu')._values().numpy()
            Lambda = coo_matrix((vals, (idxs[0, :], idxs[1, :])), dtype=NP_DTYPE)
        else:
            Lambda = self._Lambda.to('cpu').numpy()
        rewards = self._rewards.to('cpu').numpy()
        eps_0 = self._eps_0.data.to('cpu').numpy()
        # Compute q
        q, info = bicgstab(Lambda, rewards, x0=self._q.to('cpu').numpy() if self._q is not None else None, tol=1e-4)
        self._q = torch.tensor(q.reshape(-1, 1), device=DEVICE, dtype=TORCH_DTYPE)
        # Compute mu
        eps_0 = eps_0.mean(axis=1, keepdims=True)
        mu, info = bicgstab(Lambda.T, eps_0, x0=self._mu.to('cpu').numpy() if self._mu is not None else None, tol=1e-4)
        self._mu = torch.tensor(mu.reshape(-1, 1), device=DEVICE, dtype=TORCH_DTYPE)

    def policy_gradient(self):
        """
        Builds the policy gradient objective function.
        """
        self._J = -torch.mean((self._eps_0.t() + self._gamma * self._mu.t() @ self._P) @ self._q, dim=0)

    def objective_function(self):
        """
        Builds the average value function J = integral(mu_0(s)*V(s) ds), where V(s)=eps(s).T@q
        """
        with torch.no_grad():
            self._objective = torch.mean(self._eps_0.t() @ self._q, dim=0)

    def compute_graph(self):
        """
        Builds the computational graph for the policy gradient.
        """
        self.build_full_P()
        self.build_Lambda()
        self.build_eps_0()
        self.compute_q_mu()
        self.policy_gradient()
        self.objective_function()

    def fit(self, n_policy_updates=100, optimizer=None,
            eval_mdp=None, eval_every_n=50, eval_n_episodes=1, eval_initial_state=None,
            eval_transform_to_internal_state=None, eval_render=False,
            results_dir='/tmp/'):
        """
        Optimizes the parametrized policy with gradient ascent.

        :param n_policy_updates: number of policy updates
        :type n_policy_updates: int
        :param optimizer: function with torch.optim optimizer
        :type optimizer: torch.optim
        :param eval_mdp: the mdp to evaluate the policy on
        :type eval_mdp: MDP
        :param eval_every_n: evaluate the policy every n iterations of policy update
        :type eval_every_n: int
        :param eval_n_episodes: evaluate the policy n episodes in the mdp
        :type eval_n_episodes: int
        :param eval_initial_state: initial state of the evaluation mdp
        :type eval_initial_state: ndarray
        :param eval_transform_to_internal_state: function to transform initial state to the internal state of the mdp
        :type eval_transform_to_internal_state: callable function
        :param eval_render: render the environment
        :type eval_render: bool
        """
        self._iter = 0
        evaluation_params = {'n_episodes': eval_n_episodes, 'initial_state': eval_initial_state,
                             'transform_to_internal_state': eval_transform_to_internal_state, 'render': eval_render}
        n_policy_updates = n_policy_updates
        optimizer = optimizer(self._policy.parameters()) if callable(optimizer) else None
        print()
        for i in range(n_policy_updates):
            self.compute_graph()
            optimizer.zero_grad()
            self._J.backward()
            optimizer.step()

            print("Policy update {}/{}".format(i, n_policy_updates), end=" ")
            print(" | J: {:.1f} | Loss: {:.1f}".format(self._objective.item(), self._J.item()))

            if i % eval_every_n == 0 or i == n_policy_updates - 1:
                data = eval_mdp.evaluate(self._policy, **evaluation_params, results_dir=results_dir)
                _, _, rewards, _, _ = data.get_full_batch()
                rewards = rewards.reshape(-1)
                Jenv = np.sum(rewards)
                Jenv_discounted = np.sum(rewards * np.array([self._gamma**i for i in range(len(rewards))]))
                print("Jenv: {:.1f} | Jenv_discounted: {:.1f}".format(Jenv, Jenv_discounted))

            self._iter += 1
