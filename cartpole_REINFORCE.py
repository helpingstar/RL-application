import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device : {device}')
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(
        env, video_folder='./record_video')
    pi = Policy().to(device)
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        state, _ = env.reset()
        done = False

        while not done:  # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(state).float().to(device))
            prob = prob.cpu()
            m = Categorical(prob)
            action = m.sample()
            action = action.item()
            next_state, reward, done, _, info = env.step(action)
            pi.put_data((reward, prob[action]))
            state = next_state
            score += reward

        pi.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                f' | {n_epi-print_interval:05d}~{n_epi:05d} | AVG SCORE : {score/print_interval}')
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()
