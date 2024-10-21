import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import gym
from gym import spaces
import numpy as np


class myEnv(gym.Env):
    def __init__(self):
        super(myEnv, self).__init__()
        self.target = (0, 0)
        self.n_problems = 10
        self.target = (0, 0)
        self.radius = 0.5
        self.problems = [
            (np.random.normal(0.0, 1), np.random.normal(0.0, 1))
            for _ in range(self.n_problems)
        ]

        self.step_size = 0.2

        # Define the action and observation space
        # There are 4 possible actions: UP, DOWN, LEFT, RIGHT
        self.action_space = spaces.Discrete(4)

        # The observation is a 2D grid (5x5) and the agent's position
        self.observation_space = spaces.Box(low=-5, high=5, shape=(2,), dtype=int)

        # Grid dimensions
        self.grid_size = 5
        self.state = None
        self.goal = (0, 0)  # Define the goal position

    def reset(self):
        self.target = (0, 0)
        self.problems = [
            (np.random.normal(0.0, 1), np.random.normal(0.0, 1))
            for _ in range(self.n_problems)
        ]

        self.state = (
            np.random.normal(0.0, 6),
            np.random.normal(0.0, 6),
        )
        return np.array(self.state)

    def step(self, action):
        # Define the action effects
        if action == 0:  # UP
            self.state = (
                max(self.state[0] - self.step_size, 0),
                self.state[1],
            )
        elif action == 1:  # DOWN
            self.state = (
                min(self.state[0] + self.step_size, self.grid_size - 1),
                self.state[1],
            )
        elif action == 2:  # LEFT
            self.state = (
                self.state[0],
                max(self.state[1] - self.step_size, 0),
            )
        elif action == 3:  # RIGHT
            self.state = (
                self.state[0],
                min(self.state[1] + self.step_size, self.grid_size - self.step_size),
            )

        # Check if the agent has reached the goal
        done = self.state == self.goal

        # Reward is 1 if reached goal, otherwise 0
        reward = 1 if done else 0

        return np.array(self.state), reward, done, {}

    def render(self, mode="human"):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal")

        for xc, yc in self.problems:
            ax.scatter(xc, yc, s=50, color="red")
            ax.add_artist(
                plt.Circle(
                    (xc, yc),
                    radius=self.radius,
                    edgecolor="red",
                    fill=False,
                    linewidth=1,
                    label="Circle around Point",
                )
            )
            ax.add_artist(
                plt.Circle(
                    (xc, yc),
                    radius=self.radius * 3,
                    edgecolor="yellow",
                    fill=False,
                    linewidth=0.1,
                    label="Circle around Point",
                )
            )

        sc0 = ax.scatter(self.target[0], self.target[1], s=100, marker="x")
        ax.add_artist(
            plt.Circle(
                (self.target[0], self.target[1]),
                radius=self.radius * 10,
                edgecolor="blue",
                fill=False,
                linewidth=0.1,
            )
        )
        sc1 = ax.scatter(self.state[0], self.state[1], s=100, marker="o")

        self.fig = fig
        self.ax = ax
