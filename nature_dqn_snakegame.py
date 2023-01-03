from tqdm import tqdm
import matplotlib.pyplot as plt
import gym_snakegame
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# input.shape : (15, 15)
class DQN(nn.Module):
    def __init__(self,
                 observation_space: spaces.MultiBinary,
                 action_space: spaces.Discrete):
        super().__init__()
        assert len(
            observation_space.shape) == 3, 'observation space must have the form channels x width x height'
        assert type(
            action_space) == spaces.Discrete, 'action_space must be of type Discrete'
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64*7*7 , out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=action_space.n)
        )

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0],-1)
        return self.fc(conv_out)
    
import numpy as np


class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        indices = np.random.randint(0, len(self._storage) - 1, size=batch_size)
        return self._encode_sample(indices)
    
class DQNAgent:
    def __init__(self,
                 observation_space: spaces.MultiBinary,
                 action_space: spaces.Discrete,
                 replay_buffer: ReplayBuffer,
                 lr,
                 batch_size,
                 gamma,
                 device=torch.device("cpu" )):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """

        self.memory = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma

        self.policy_network = DQN(observation_space, action_space).to(device)
        self.target_network = DQN(observation_space, action_space).to(device)
        self.update_target_network()
        self.target_network.eval()

        self.optimiser = torch.optim.RMSprop(self.policy_network.parameters()
            , lr=lr)        
        ## self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

        self.device = device

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        device = self.device

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values, _ = next_q_values.max(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        input_q_values = self.policy_network(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(input_q_values, target_q_values)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        del states
        del next_states
        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        device = self.device
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_network(state)
            _, action = q_values.max(1)
            return action.item()

def linear_epsilon_decrease(now_step, total_step, max_eps=1, min_eps=0.1, ratio=0.1):
    eps_timesteps = ratio * float(total_step)
    fraction = min(1.0, float(now_step) / eps_timesteps)
    eps = max_eps + fraction * (min_eps - max_eps)
    return eps

class PyTorchDimension(gym.ObservationWrapper):
    """Image shape to num_channels x height x width"""

    def __init__(self, env):
        super(PyTorchDimension, self).__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=(1, shape[0], shape[1]), dtype=np.uint8)

    def observation(self, observation):
        shape = observation.shape
        return observation.reshape(1, shape[0], shape[1])
    
EPISODE_NUM = 5000
SEED_NUM = 20
TOTAL_STEP = 1000000

MAX_EPSILON = 1.0
MIN_EPSILON = 0.1
env = gym.make('gym_snakegame/SnakeGame-v0', size=15, n_target=1, render_mode='human')
env = PyTorchDimension(env)

replay_buffer = ReplayBuffer(30000)

agent = DQNAgent(env.observation_space,
                 env.action_space,
                 replay_buffer,
                 lr=1e-3,
                 batch_size=64,
                 gamma=0.99,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

state, _ = env.reset()
score = []
episode_rewards = [0.0]
step_count = 0
for t in tqdm(range(TOTAL_STEP)):
    sample = random.random()
    epsilon = linear_epsilon_decrease(t, TOTAL_STEP, max_eps=1, min_eps=0.1, ratio=0.5)
    if sample > epsilon:
        # Exploit
        action = agent.act(state)
    else:
        # Explore
        action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    agent.memory.add(next_state, action, reward, next_state, float(terminated))
    state = next_state
    episode_rewards[-1] += reward

    if terminated:
        state, _ = env.reset()
        episode_rewards.append(0.0)
        if len(episode_rewards) % 50 == 0:
            print(sum(episode_rewards[-10:])/10)

    if t > 10000:
        agent.optimise_td_loss()

    if t > 10000 and t % 500 == 0:
        agent.update_target_network()