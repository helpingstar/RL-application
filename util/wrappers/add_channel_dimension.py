import gymnasium as gym
import numpy as np

class AddChannelDimension(gym.ObservationWrapper):
    """Image shape to num_channels x height x width"""

    def __init__(self, env):
        super(AddChannelDimension, self).__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=(1, shape[0], shape[1]), dtype=np.uint8)

    def observation(self, observation):
        shape = observation.shape
        return observation.reshape(1, shape[0], shape[1])