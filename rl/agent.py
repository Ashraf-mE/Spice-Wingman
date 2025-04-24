import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import MultivariateNormal
from utils.config import PPO_LR, PPO_GAMMA, PPO_EPS_CLIP, PPO_K_EPOCHS

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        action_mean = self.actor(state)
        state_value = self.critic(state)
        return action_mean, state_value

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=PPO_LR)
        self.cov_matrix = torch.diag(torch.full((action_dim,), 0.5))
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_mean, _ = self.policy(state_tensor)
        dist = MultivariateNormal(action_mean, self.cov_matrix)
        action = dist.sample()
        return action.detach().numpy().flatten()
    
    def update(self, memory):
        states, actions, rewards, next_states, dones = memory.get_batches()
        for _ in range(PPO_K_EPOCHS):
            action_means, state_values = self.policy(states)
            dist = MultivariateNormal(action_means, self.cov_matrix)
            log_probs = dist.log_prob(actions)
            advantages = rewards + PPO_GAMMA * state_values.squeeze() * (1 - dones) - state_values.squeeze()
            loss = -(log_probs * advantages).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()