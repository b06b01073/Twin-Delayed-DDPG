import gym
from argparse import ArgumentParser
import torch
import numpy as np

from TD3_agent import TD3Agent
from utils import plot_result

import os

if __name__ == '__main__':
    # read and parse arguments
    parser = ArgumentParser()
    parser.add_argument('--max_steps', '-s', type=int, default=1000000)
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--exploration_mu', type=float, default=0)
    parser.add_argument('--exploration_sigma', type=float, default=0.1)
    parser.add_argument('--smoother_mu', type=float, default=0)
    parser.add_argument('--smoother_sigma', type=float, default=0.2)
    parser.add_argument('--smoother_clip', type=float, default=0.5)
    parser.add_argument('--delay', '-d', type=int, default=2)
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--gamma', '-g', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--tau', '-t', type=float, default=5e-3)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--warmup', '-w', action='store_true') # warmup will fill the replay buffer by transitions with random actions before training
    parser.add_argument('--warmup_step', default=20000,type=int)   
    parser.add_argument('--save_dir', type=str, default='./model_params')
    args = parser.parse_args()


    # create necessary dirs
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    save_path = os.path.join('./result', args.env_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    

    # model device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')

    trials_avg_rewards = []


    # train model and repeat for `trials` times
    for trial in range(args.trials):
        env = gym.make(args.env_name)
        seed = args.seed + trial

        agent = TD3Agent(env.observation_space, env.action_space, args, device, seed)
        avg_rewards = agent.do_task(env, args.max_steps)
        trials_avg_rewards.append(avg_rewards)

        # save record as numpy binary (so that you can stop training and come back later)
        np.save(os.path.join(save_path, f'{seed}_avg_reward'), avg_rewards)

    plot_result(trials_avg_rewards, args.max_steps, args.eval_freq, args.env_name, save_path)