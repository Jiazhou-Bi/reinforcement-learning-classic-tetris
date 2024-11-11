# installing packages

import gym
import gym_tetris
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# setting up environment
env = gym.make('TetrisA-v0')

# defining the deep QNet to be used in this model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
# defining the agent
epsilon = 1
epsilon_decay = 0.995
epsilon_min = 0.01
gamma = 0.99
learning_rate = 0.001
memory = deque(maxlen=2000)
batch_size = 64

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.memory = memory

        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0) #this part is not fuuly understood
        act_values = self.model(state)
        return torch.argmax(act_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += gamma * torch.max(self.model(torch.FloatTensor(next_state)))

            target_f =self.model(torch.FloatTensor(state))
            target_f[action] = target
            
            self.optimizer.zerro_grad()
            loss = self.criterion(target_f, self.mode(torch.FloatTensor(state)))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay