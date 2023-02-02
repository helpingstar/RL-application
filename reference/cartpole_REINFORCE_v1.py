# source : https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py

import gymnasium as gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


args = {
    'log_interval' : 10,
    'seed' : 543,
    'gamma' : 0.99,
    'learning_rate' : 0.003
}

env = gym.make('CartPole-v1')
env.reset(seed=args['seed'])
torch.manual_seed(args['seed'])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=args['learning_rate'])
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    # probs : π(a|s)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    # logπ(a|s)
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        # returns_t = G_t = r_1 + γr_2 + ... + γ^{t-1}r_t
        R = r + args['gamma'] * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    # torch.cat : [tensor([a]), tensor([b]), tensor([c]), ...] -> tensor([a, b, c, d])
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # RR = 0.05 * ER + 0.95 * RR
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args['log_interval'] == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')
        if running_reward > env.spec.reward_threshold:
            print(f"Solved! Running reward is now {running_reward:.3f} and "
                  "the last episode runs to {t} time steps!")
            break


if __name__ == '__main__':
    main()
