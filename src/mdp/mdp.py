import numpy as np
import torch
from sklearn.utils.extmath import cartesian

from src.dataset.dataset import Dataset
from src.utils.utils import DEVICE, TORCH_DTYPE


class MDP:

    def __init__(self, env):
        """
        General class to sample and evaluate a policy on a CONTINUOUS mdp.

        :param env: gym like environment initialized with gym.make('Environment-v0')
        """
        self._env = env
        self.s_dim = np.prod(env.observation_space.shape)
        self.a_dim = np.prod(env.action_space.shape)
        self._quanser_robots = False
        try:
            if 'quanser_robots' in str(type(env.dyn)):
                self._quanser_robots = True
        except AttributeError:
            pass

    def get_samples(self, sampling_type=None, states=None, actions=None, transform_to_internal_state=lambda x: x,
                    policy=None, n_samples=None, n_trajectories=None, initial_state=None, max_ep_transitions=100,
                    render=False, press_enter_to_start=False, dataset=None, results_dir='/tmp/'):
        """
        Collect samples from the mdp.
        If sampling_type is 'uniform', provide the grid of states and actions to samples uniformly from, and
        additionally transform_to_internal_state function.
        If sampling_type is 'behavioral', provide at least the behavioral policy, and no states and actions.
        NOTE: opt between either n_samples or n_trajectories. In the case of n_trajectories specify the maximum number
        of transitions per episode max_ep_transitions.

        :param sampling_type: can be uniform or behavioral
        :type sampling_type: str
        :param states: a batch of states to sample on
        :type states: ndarray
        :param actions: a batch of actions to sample each state on
        :type actions: ndarray
        :param transform_to_internal_state: transform the states or initial state to an internal state of the mdp
        :type transform_to_internal_state: function
        :param policy: behavioral policy outputing an action given a state
        :type policy: callable function
        :param n_samples: number of samples to collect (EXCLUSIVE between n_samples and n_trajectories)
        :type n_samples: int
        :param n_trajectories: number of trajectories to collect (EXCLUSIVE between n_samples and n_trajectories)
        :type n_trajectories: int
        :param initial_state: initial state to start sampling from (e.g., in the pendulum we want to start from the
            upright position)
        :type initial_state: ndarray
        :param max_ep_transitions: maximum number of transitions before reseting to the initial state
        :type max_ep_transitions: int
        :param render: render the environment
        :type render: int
        :param press_enter_to_start: wait for user input to start collecting trajectories
        :type press_enter_to_start: bool
        :param dataset: if a dataset is provided add trajectories to it, else start an empty one
        :type dataset: Dataset
        :param results_dir: provide a directory path to save intermediate results
        :type results_dir: str
        :return: a dataset containing the collected trajectories
        """
        dataset = dataset if dataset is not None else Dataset(results_dir=results_dir)
        if sampling_type == 'uniform':
            for state in states:
                for action in actions:
                    self._env.reset()
                    if self._quanser_robots:
                        self._env.env._sim_state = np.copy(transform_to_internal_state(state))
                    else:
                        self._env.env.state = np.copy(transform_to_internal_state(state))
                    state_next, rew, done, _ = self._env.step(action)
                    dataset.add_trajectory([(state, action, rew, state_next, done)])
        elif sampling_type == 'behavioral':
            behavioral_policy = policy
            i = 0
            n_samples_collected = 0
            criteria = 0
            if n_samples is not None:
                criteria = n_samples
            elif n_trajectories is not None:
                criteria = n_trajectories
            while i < criteria:
                if press_enter_to_start:
                    if render:
                        self._env.render()
                    input("Press ENTER to start")
                state = self._env.reset()
                if initial_state is not None:
                    state = initial_state
                    if self._quanser_robots:
                        self._env.env._sim_state = np.copy(transform_to_internal_state(state))
                    else:
                        self._env.env.state = np.copy(transform_to_internal_state(state))
                trajectory = []
                for j in range(max_ep_transitions):
                    if render:
                        self._env.render()
                    action = behavioral_policy(x=state)
                    state_next, rew, done, _ = self._env.step(action)
                    trajectory.append((state, action, rew, state_next, done))
                    n_samples_collected += 1
                    print("\r{}/{} - {}".format(i + 1, criteria, n_samples_collected), end=' ', flush=True)
                    if n_samples is not None:
                        i += 1
                    if done or (i >= n_samples if n_samples is not None else False):
                        break
                    state = state_next
                dataset.add_trajectory(trajectory)
                if n_trajectories is not None:
                    i += 1
                    if i >= n_trajectories:
                        break
        self._env.close()
        return dataset

    def evaluate(self, policy, n_episodes=1, initial_state=None, transform_to_internal_state=None, render=False,
                 results_dir='/tmp/'):
        """
        Evaluate a policy on the mdp.

        :param policy: a policy outputing an action
        :type policy: Policy
        :param n_episodes: number of episodes to evaluate
        :type n_episodes: int
        :param initial_state: the initial state to evaluate from
        :type initial_state: int
        :param transform_to_internal_state: transform the states or initial state to an internal state of the mdp
        :type transform_to_internal_state: function
        :param render: reder the environment
        :type render: bool
        :return: a Dataset with the trajectories
        """
        dataset = Dataset(results_dir=results_dir)
        for episode in range(n_episodes):
            trajectory = []
            state = self._env.reset()
            if initial_state is not None:
                state = initial_state
                if self._quanser_robots:
                    self._env.env._sim_state = np.copy(transform_to_internal_state(state))
                else:
                    self._env.env.state = np.copy(transform_to_internal_state(state))
            for j in range(self._env._max_episode_steps):
                with torch.no_grad():
                    state = torch.tensor(state, device=DEVICE, dtype=TORCH_DTYPE)
                    action = policy(state).to('cpu').numpy().reshape((-1,))
                    state_next, rew, done, _ = self._env.step(action)
                    trajectory.append((state.to('cpu').numpy(), action, rew, state_next, done))
                    state = state_next
                    if render:
                        self._env.render()
                    if done:
                        break
            dataset.add_trajectory(trajectory)
        self._env.close()
        return dataset

    def discretize_space(self, space='state', levels=None):
        """
        Discretize the state or action space with granularity based on levels, and create the cartesian product of the
        referred space.
        The Pendulum environment is treated as a special case where the original state space is 3-dimensional
        (cos, sin, angular velocity), but we specify the levels for (angle, angular velocity).

        :param space: 'state' or 'action' space
        :type space: str
        :param levels: discretization granularity per space dimension
        :type levels: list
        :return: a discretized space with granularity based on levels
        """
        discretized = []
        for i, d in enumerate(levels):
            low, high = None, None
            if space == 'state':
                if self._env.unwrapped.spec.entry_point == 'gym.envs.classic_control:PendulumEnv':
                    # Special discretization for the Pendulum. Instead of discretizing cos(theta), sin(theta), do it
                    # directly in theta
                    if i == 0:
                        low = -np.pi
                        high = np.pi
                    elif i == 1:
                        low = self._env.observation_space.low[2]
                        high = self._env.observation_space.high[2]
                else:
                    low = self._env.observation_space.low[i]
                    high = self._env.observation_space.high[i]
            elif space == 'action':
                low = self._env.action_space.low[i]
                high = self._env.action_space.high[i]

            discretized.append(np.linspace(low, high, num=d))

        discretized = cartesian(discretized)
        if self._env.unwrapped.spec.entry_point == 'gym.envs.classic_control:PendulumEnv' and space == 'state':
            # Special discretization for the Pendulum state. Instead of discretizing cos(theta), sin(theta), do it
            # directly in theta
            discretized_augment = np.zeros((discretized.shape[0], 3))
            discretized_augment[:, 0] = np.cos(discretized[:, 0])
            discretized_augment[:, 1] = np.sin(discretized[:, 0])
            discretized_augment[:, 2] = discretized[:, 1]
            discretized = discretized_augment

        return discretized
