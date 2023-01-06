import gymnasium as gym
from gymnasium import spaces



def obs_space_to_shape(obs_space: spaces.Space):
    if isinstance(obs_space, spaces.Box):
        return obs_space.shape
    if isinstance(obs_space, spaces.Discrete):
        return (1,)

    else:
        raise NotImplementedError(f"{obs_space} observation space is not supported")
