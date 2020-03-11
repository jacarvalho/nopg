import numpy as np
import gym
from gym.envs import register

from quanser_robots import GentlyTerminating
from quanser_robots.qube import SwingUpCtrl
from src.dataset.dataset import Dataset


register(
    id='Qube-100-v1',
    entry_point='quanser_robots.qube.qube:Qube',
    max_episode_steps=1500,
    kwargs={'fs': 200.0, 'fs_ctrl': 200.0}
)

env = GentlyTerminating(gym.make('Qube-100-v1'))

dataset = Dataset()
n_trajectories = 10
for traj in range(n_trajectories):
    trajectory = []
    ctrl = SwingUpCtrl()
    obs = env.reset()
    done = False
    while not done:
        env.render()
        act = ctrl(obs)
        obs_n, r, done, _ = env.step(act)
        trajectory.append((obs, act, r, obs_n, done))
        if done:
            break
        obs = np.copy(obs_n)
    env.close()
    dataset.add_trajectory(trajectory)

dataset.update_dataset_internal()

filename = "../datasets/qube/{}_trajectories.npy".format(n_trajectories)
dataset.save_trajectories_to_file(filename)

