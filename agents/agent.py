import random
import numpy as np
from sklearn.linear_model import LinearRegression


def min_max_restriction(v, min_v, max_v):
    return max(min(v, max_v), min_v)


class Agent:
    def __init__(self, name):
        self.name = name
        self.exploration_rate = 0.1
        self.health = 100
        self.target = (0.0, 0.0)
        self.model = None

    def choose_action(self, state):
        raise NotImplementedError

    def learn(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def reset(self):
        self.health = 100


class AgentAttack(Agent):

    def choose_action(self, state):
        features = sum([list(state), list(self.target), [1]], [])
        mu_x, std_x, mu_y, std_y, speed = self.model.predict([features])[0]
        mu_x, std_x, mu_y, std_y, speed = (
            min_max_restriction(round(mu_x, 3), -0.1, 0.1),
            min_max_restriction(round(std_x, 3), -0.1, 0.1),
            min_max_restriction(round(mu_y, 3), -0.1, 0.1),
            min_max_restriction(round(std_y, 3), -0.1, 0.1),
            min_max_restriction(round(speed, 3), -0.1, 0.1),
        )

        if random.uniform(0, 1) < self.exploration_rate:
            step_x = np.random.normal(0.0, 0.5)
            step_y = np.random.normal(0.0, 0.5)
        else:
            step_x = np.random.normal(mu_x, max(0.001, std_x))
            step_y = np.random.normal(mu_y, max(0.001, std_y))

        next_x = state[0] + step_x * speed
        next_y = state[1] + step_y * speed

        return next_x, next_y, (mu_x, std_x, mu_y, std_y, speed)

    def warmup_model(self):
        # do warmup
        X = np.random.rand(300, 5)  # x, y, tx, ty, reward
        y = np.random.rand(300, 5)  # mu_x, std_x, mu_y, std_y, speed
        self.model = LinearRegression().fit(X, y)

    def learn(self, state, action, reward, next_state, done):
        X = [sum([list(state), list(self.target), [reward]], [])]
        y = [list(action)]
        self.model.fit(X, y)


class AgentDefend(Agent):

    def choose_action(self, state):
        x1, y1, x2, y2, target = state
        x2, y2 = target

        if random.uniform(0, 1) < self.exploration_rate:
            d = (np.random.choice([1, -1]), np.random.choice([1, -1]))
        else:
            d = (1.0 if (x2 - x1) > 0 else -1.0, 1.0 if (y2 - y1) > 0 else -1.0)

        next_x = x1 + self.speed * d[0]
        next_y = y1 + self.speed * d[1]

        return next_x, next_y
