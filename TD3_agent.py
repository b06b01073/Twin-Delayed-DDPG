import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import gym
from tqdm import tqdm

from model import Actor, Critic
from noise_generator import GaussianNoise
from replay_buffer import ReplayMemory  


class TD3Agent:
    def __init__(self, obs_space, action_space, args, device, seed):
        self.obs_dim = obs_space.shape
        self.action_dim = action_space.shape

        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_low_tensor = torch.from_numpy(action_space.low).float().to(device)
        self.action_high_tensor = torch.from_numpy(action_space.high).float().to(device)


        self.critic1 = Critic(self.obs_dim[0], self.action_dim[0]).to(device)
        self.critic2 = Critic(self.obs_dim[0], self.action_dim[0]).to(device)
        self.target_critic1 = Critic(self.obs_dim[0], self.action_dim[0]).to(device)
        self.target_critic2 = Critic(self.obs_dim[0], self.action_dim[0]).to(device)

        self.actor = Actor(self.obs_dim[0], self.action_dim[0]).to(device)
        self.target_actor = Actor(self.obs_dim[0], self.action_dim[0]).to(device)

        self.hard_update(self.target_critic1, self.critic1)
        self.hard_update(self.target_critic2, self.critic2)
        self.hard_update(self.target_actor, self.actor)

        self.critic1_optim = optim.Adam(params=self.critic1.parameters(), lr=args.lr)
        self.critic2_optim = optim.Adam(params=self.critic2.parameters(), lr=args.lr)
        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=args.lr)
        self.mse_loss = nn.MSELoss()

        self.action_noise = GaussianNoise(size=self.action_dim, mu=args.exploration_mu, sigma=args.exploration_sigma)
        self.policy_smoother = GaussianNoise(size=self.action_dim, mu=args.smoother_mu, sigma=args.smoother_sigma, clip=args.smoother_clip)
        self.random_action_generator = GaussianNoise(size=self.action_dim, mu=self.warmup_mu, sigma=self.warmup_sigma)


        self.memory = ReplayMemory(capacity=args.max_steps) # the entire history of the agent
        self.buffer_capacity = args.max_steps
        self.batch_size = args.batch_size
        self.device = device

        self.steps = 0 # for actor and target critic update
        self.eval_freq = args.eval_freq
        self.gamma = args.gamma
        self.delay = args.delay
        self.tau = args.tau
        self.seed = seed
        self.env_name = args.env_name
        self.eval_episodes = args.eval_episodes

        self.warmup = args.warmup
        self.warmup_mu = args.warmup_mu
        self.warmup_sigma = args.warmup_sigma

    def select_action(self, obs, enable_noise=True):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device)
            action = self.actor(obs).cpu().detach().numpy()

            if enable_noise:
                action = np.clip(action + self.action_noise.sample(), a_min=self.action_low, a_max=self.action_high)
            return action

    
    def hard_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)


    def evaluate(self):
        env = gym.make(self.env_name)

        avg_reward = 0.
        for i in range(self.eval_episodes):
            obs = env.reset()
            total_reward = 0
            while True:
                action = self.select_action(obs, enable_noise=False)
                obs, reward, terminated, _ = env.step(action)
                avg_reward += reward
                total_reward += reward

                if terminated:
                    break
            print(f'total reward of eval episode {i}: {total_reward}')

        avg_reward /= self.eval_episodes

        return avg_reward

    def do_warmup(self, env):
        obs = env.reset(seed=self.seed)
        
        for _ in tqdm(range(self.buffer_capacity), desc='warming up...'):
            action = np.clip(self.random_action_generator.sample(), a_max=self.action_high, a_min=self.action_low)
            next_obs, reward, terminated, _ = env.step(action)

            self.memory.append(obs, action, [reward], next_obs, [int(terminated)])
            obs = next_obs

            if terminated:
                obs = env.reset(seed=self.seed)


    def do_task(self, env, max_steps):
        
        if self.warmup:
            self.do_warmup(env)


        obs = env.reset(seed=self.seed)
        avg_rewards = [self.evaluate()] # evaluate the init model
        for i in range(max_steps):

            action = self.select_action(obs)
            next_obs, reward, terminated, _ = env.step(action)

            self.memory.append(obs, action, [reward], next_obs, [int(terminated)])


            if (i + 1) % self.eval_freq == 0:
                print(f'evaluating...(steps: {i + 1})')

                with torch.no_grad():
                    avg_reward = self.evaluate()
                    avg_rewards.append(avg_reward)
                    print(f'avg_reward: {avg_reward}')

            self.steps += 1
            obs = next_obs
            self.learn()

            if terminated:
                obs = env.reset(seed=self.seed)

        return avg_rewards

    def learn(self):

        # sample experience from replay buffer
        experience = self.memory.sample(self.batch_size, self.device)
        if experience is None:
            return
        
        obs, action, reward, next_obs, terminated = experience


        # build q target
        with torch.no_grad():
            next_action = self.target_actor(next_obs) 
            next_action += torch.from_numpy(self.policy_smoother.sample()).float().to(self.device)
            next_action = torch.clamp(next_action, min=self.action_low_tensor, max=self.action_high_tensor)

            q_target = reward + self.gamma * (1 - terminated) * torch.min(self.target_critic1(next_obs, next_action), self.target_critic2(next_obs, next_action))

        # update critic1
        q_pred1 = self.critic1(obs, action)
        critic_loss = self.mse_loss(q_pred1, q_target.detach())
        self.critic1_optim.zero_grad()
        critic_loss.backward()
        self.critic1_optim.step()

        # update critic2
        q_pred2 = self.critic2(obs, action)
        critic_loss = self.mse_loss(q_pred2, q_target.detach())
        self.critic2_optim.zero_grad()
        critic_loss.backward()
        self.critic2_optim.step()


        # update actor and target network
        if self.steps % self.delay == 0:

            # update actor
            grad = -self.critic1(obs, self.actor(obs)).mean() # negative sign for gradient ascend
            self.actor_optim.zero_grad()
            grad.backward()
            self.actor_optim.step()


            # soft update target network
            for source_param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

            for source_param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

            for source_param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)


        
