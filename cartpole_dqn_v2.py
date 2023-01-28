from tqdm import tqdm
import gymnasium as gym
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import random
import time
import wandb

from torch.utils.tensorboard import SummaryWriter
from util.scheduler.linear_schedule import linear_schedule
from util.buffer.ReplayBuffer import ReplayBuffer
args = {
    'env_id': 'CartPole-v1',
    'algorithm': 'DQN',
    'algorithm_version': 'v2',
    'truncated' : 500,
    'seed': 42,
    'cuda': True,
    'learning_rate' : 0.0003,
    'buffer_size' : 10000,
    'total_timesteps' : 300000,
    'start_e' : 1,
    'end_e' : 0.01,
    'exploration_fraction' : 0.5,
    'wandb_entity' : None,
    'learning_starts' : 10000,
    'train_frequency' : 1,
    'batch_size' : 128,
    'target_network_frequency' : 500,
    'gamma' : 0.99,
    'capture_video' : False,
    'loss_function' : 'smooth_l1_loss',
    'grad_clipping' : 10.0
    }

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n)
        )

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    project_path = args['env_id'].split('/')[-1]
    device = torch.device("cuda" if torch.cuda.is_available() and args["cuda"] else "cpu")
    run_name=f"{args['algorithm']}_{args['algorithm_version']}_{int(time.time())}"

    print(f'project_path: {project_path}, device : {device}, run_name : {run_name}')
    wandb.init(
        # set the wandb project where this run will be logged
        name=run_name,
        project=project_path,
        entity=args['wandb_entity'],
        # sync_tensorboard=True,
        config=args,
        monitor_gym=True,
        save_code=True
    )

    writer = SummaryWriter(f'runs/{project_path}/{run_name}')
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in args.items()])),
    )

    env = gym.make(args["env_id"], render_mode='rgb_array')
    if args['truncated']:
        env = gym.wrappers.TimeLimit(env, args['truncated'])
    env = gym.wrappers.AutoResetWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    q_network = QNetwork(env).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args['learning_rate'])
    target_network = QNetwork(env).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        env.observation_space,
        args['buffer_size'],
        args['batch_size']
    )

    start_time = time.time()

    obs, _ = env.reset()
    for global_step in tqdm(range(args['total_timesteps'])):
        epsilon = linear_schedule(args['start_e'],
                                args['end_e'],
                                args['exploration_fraction'] * args['total_timesteps'],
                                global_step)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            action = torch.argmax(q_values).item()

        next_obs, reward, terminate, truncate, info = env.step(action)
        rb.store(obs, action, reward, next_obs, terminate)

        obs = next_obs

        if 'episode' in info.keys():
            writer.add_scalar("charts/episodic_return", info['episode']['r'], global_step)
            wandb.log({"charts/episodic_return": info['episode']['r']}, step=global_step)

        writer.add_scalar("charts/epsilon", epsilon, global_step)
        wandb.log({"charts/epsilon": epsilon}, step=global_step)

        if global_step > args['learning_starts']:
            if global_step % args['train_frequency'] == 0:

                samples = rb.sample_batch()
                states = torch.FloatTensor(samples['obs']).to(device)
                next_states = torch.FloatTensor(samples['next_obs']).to(device)
                actions = torch.LongTensor(samples['acts']).reshape(-1, 1).to(device)
                rewards = torch.FloatTensor(samples['rews']).reshape(-1, 1).to(device)
                dones = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

                with torch.no_grad():
                    target_max, _ = target_network(next_states).max(dim=1)
                    td_target = rewards.flatten() + args['gamma'] * target_max * (1 - dones.flatten())
                old_val = q_network(states).gather(1, actions).squeeze()

                if args['loss_function'] == 'mse_loss':
                    loss = F.mse_loss(td_target, old_val)
                elif args['loss_function'] == 'smooth_l1_loss':
                    loss = F.smooth_l1_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    wandb.log({"losses/td_loss": loss,
                            "losses/q_values": old_val.mean().item(),
                            "charts/SPS": int(global_step / (time.time() - start_time))}, step=global_step)
                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                if args['grad_clipping'] is not None:
                    clip_grad_norm_(q_network.parameters(), args['grad_clipping'])
                optimizer.step()

            # update the target network
            if global_step % args['target_network_frequency'] == 0:
                target_network.load_state_dict(q_network.state_dict())
    env.close()
    writer.close()
    wandb.finish()

    if not os.path.isdir(f"weights/{project_path}"):
        os.makedirs(f"weights/{project_path}")

    torch.save(q_network, f"weights/{project_path}/{run_name}_q_network.pt")
    torch.save(target_network, f"weights/{project_path}/{run_name}_target_network.pt")
