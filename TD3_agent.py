import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from model import Actor, Critic
from noise_generator import GaussianNoise
from replay_buffer import ReplayMemory  

class TD3Agent:
    def __init__(self, obs_space, action_space, args, device):
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


        self.memory = ReplayMemory(capacity=args.capacity)
        self.batch_size = args.batch_size
        self.device = device

        self.steps = 0 # for actor and target critic update
        self.gamma = args.gamma
        self.delay = args.delay
        self.tau = args.tau


    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device)
            action = self.actor(obs).cpu().detach().numpy()
            action = np.clip(action + self.action_noise.sample(), a_min=self.action_low, a_max=self.action_high)
            return action

    
    def hard_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)


    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)


    def do_task(self, env, episodes):
        for i in range(episodes):
            obs = env.reset()
            total_reward = 0

            while True:
                action = self.select_action(obs)
                next_obs, reward, terminated, _ = env.step(action)

                self.memory.append(obs, action, [reward], next_obs, [int(terminated)])

                self.learn()

                obs = next_obs
                if terminated:
                    break

                self.steps += 1
                total_reward += reward

            print(f'Episode {i}, reward: {total_reward}')


    def learn(self):
        experience = self.memory.sample(self.batch_size, self.device)
        if experience is None:
            return
        
        obs, action, reward, next_obs, terminated = experience

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
            grad = -self.critic1(obs, self.actor(obs)).mean() # negative sign for gradient ascend
            self.actor_optim.zero_grad()
            grad.backward()
            self.actor_optim.step()

            for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        
