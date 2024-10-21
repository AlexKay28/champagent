import random
import numpy as np
from sklearn.linear_model import LinearRegression


def min_max_restriction(v, min_v, max_v):
    return max(min(v, max_v), min_v)


class QLearningAgent:
    def __init__(
        self,
        action_space,
        state_space,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        # Q-table initialization
        self.state_space_size = state_space
        self.action_space = action_space

        self.q_table = np.zeros(state_space + (action_space.n,))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.epsilon_min = epsilon_min  # Minimum epsilon

    def discretize_state(self, observation, bins):
        # Discretize continuous state space into buckets (needed for environments like CartPole)
        lower_bounds = [-10, -10]  # Lower bounds of the state space
        upper_bounds = [10, 10]  # Upper bounds of the state space

        ratios = [observation[i] for i in range(len(observation))]
        # print(observation, ratios)
        new_state = [
            int(np.digitize(ratios[i], bins=bins[i])) for i in range(len(observation))
        ]
        return tuple(new_state)

    def choose_action(self, state):
        # Epsilon-greedy policy for action selection
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(
                [i for i in range(self.action_space.n)]
            )  # Exploration: Random action
        else:
            return np.argmax(
                self.q_table[state]
            )  # Exploitation: Choose the best action

    def update_q_value(self, state, action, reward, next_state):
        # Q-learning formula to update the Q-table
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_delta

    def decay_epsilon(self):
        # Gradually decrease epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
