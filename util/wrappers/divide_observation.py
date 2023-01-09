import numpy as np
import gymnasium as gym


class RelativePosition(gym.ObservationWrapper):
    def __init__(self, env, divide=1.0):
        super().__init__(env)
        self.divide = divide

    def observation(self, obs):
        return obs / self.divide
