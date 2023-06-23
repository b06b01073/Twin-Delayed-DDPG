import gym
from argparse import ArgumentParser
import torch

from TD3_agent import TD3Agent

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_steps', '-s', type=int, default=1000000)
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--exploration_mu', type=float, default=0)
    parser.add_argument('--exploration_sigma', type=float, default=0.1)
    parser.add_argument('--smoother_mu', type=float, default=0)
    parser.add_argument('--smoother_sigma', type=float, default=0.2)
    parser.add_argument('--smoother_clip', type=float, default=0.5)
    parser.add_argument('--delay', '-d', type=int, default=2)
    parser.add_argument('--capacity', '-c', type=int, default=100000)
    parser.add_argument('--batch_size', '-b', type=int, default=100)
    parser.add_argument('--gamma', '-g', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tau', '-t', type=float, default=5e-3)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    

    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = gym.make(args.env_name)
    agent = TD3Agent(env.observation_space, env.action_space, args, device)
    agent.do_task(env, args.max_steps)