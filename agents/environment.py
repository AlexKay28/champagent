import gym
from gym import spaces
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


from utils import dist


class myEnv(gym.Env):
    def __init__(self, map_size, n_problems):
        super(myEnv, self).__init__()
        self.target = (0, 0)
        self.n_problems = n_problems
        self.target = (0, 0)
        self.radius = 0.3
        self.step_size = 0.2

        self.problems = [
            (np.random.normal(0.0, 4), np.random.normal(0.0, 4))
            for _ in range(self.n_problems)
        ]
        self.target = (0, 0)

        # Define the action and observation space
        # There are 4 possible actions: UP, DOWN, LEFT, RIGHT
        self.action_space = spaces.Discrete(4)

        # The observation is a 2D grid (5x5) and the agent's position
        self.observation_space = spaces.Box(
            low=-map_size, high=map_size, shape=(2,), dtype=int
        )

        # Grid dimensions
        self.grid_size = 5
        self.state = None
        self.init_goal = (0, 0)  # Define the goal position
        self.init_user = (5, 5)
        self.init_counter = 1000
        self.eps = 1

    def reset(self):
        self.counter = self.init_counter
        self.state = self.init_user
        return np.array(self.state), None

    def step(self, action):
        # Define the action effects
        if action == 0:  # UP
            self.state = (
                max(self.state[0] - self.step_size, 0),
                self.state[1],
            )
        elif action == 1:  # DOWN
            self.state = (
                min(self.state[0] + self.step_size, self.grid_size - self.step_size),
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

        self.counter -= 1

        # reward
        reward = 0

        done = False
        for p in self.problems:
            if dist(self.state, p) < self.radius:
                reward -= 100

        if self.counter <= 0:
            done = True
            reward = -0.01
        if dist(self.state, self.target) < self.radius:
            done = True
            reward = 1000 * self.counter / self.init_counter

        return np.array(self.state), reward, done, {}, None

    def render(self, mode="human"):
        return
