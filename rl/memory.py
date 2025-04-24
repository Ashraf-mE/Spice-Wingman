import torch
import numpy as np

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def store(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def get_batches(self):
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        rewards = torch.FloatTensor(np.array(self.rewards))
        next_states = torch.FloatTensor(np.array(self.next_states))
        dones = torch.FloatTensor(np.array(self.dones))
        self.clear()
        return states, actions, rewards, next_states, dones
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
