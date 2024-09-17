import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DuelCNN(nn.Module):
    """
    CNN with Duel Algo. https://arxiv.org/abs/1511.06581
    """
    def __init__(self, h, w, output_size):
        super(DuelCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4,  out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convw, convh = self.conv2d_size_calc(w, h, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=3, stride=1)

        linear_input_size = convw * convh * 64  # Last conv layer's out sizes

        # Action layer
        self.Alinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Alrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Alinear2 = nn.Linear(in_features=128, out_features=output_size)

        # State Value layer
        self.Vlinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Vlrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Vlinear2 = nn.Linear(in_features=128, out_features=1)  # Only 1 node

    def conv2d_size_calc(self, w, h, kernel_size=5, stride=2):
        """
        Calcs conv layers output image sizes
        """
        next_w = (w - (kernel_size - 1) - 1) // stride + 1
        next_h = (h - (kernel_size - 1) - 1) // stride + 1
        return next_w, next_h

    def forward(self, x):
        x = x.unsqueeze(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten every batch

        Ax = self.Alrelu(self.Alinear1(x))
        Ax = self.Alinear2(Ax)  # No activation on last layer

        Vx = self.Vlrelu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx)  # No activation on last layer

        q = Vx + (Ax - Ax.mean())

        return q


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ReplayMemory:
    def __init__(self, capacity, batch_size):
        self.memory = []
        self.capacity = capacity
        self.batch_size = batch_size

    def push(self, batch):
        if self.capacity < len(self.memory):
            self.memory.pop(0)
            self.memory.append(batch)
        else:
            self.memory.append(batch)

    def not_enogh(self):
        return len(self.memory) < self.batch_size
    
    def getBatch(self):
        return random.sample(self.memory, self.batch_size)
    
class Agent:
    def __init__(self, state_dim, action_dim, mem_capacity = 2048, batch_size = 64, epsilon_decay = 0.99, epsilon_min = 0.01, lr = 0.5, gamma = 0.9):
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.gamma = gamma
        self.action_dim = action_dim
        self.policy = DQN(state_dim, action_dim)
        self.target = DQN(state_dim, action_dim)
        self.target.load_state_dict(self.policy.state_dict())
        self.replay_memory = ReplayMemory(mem_capacity, batch_size)
        self.loss = []
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        else:
            m = torch.mean(self.policy(torch.tensor(state)), dim=0)
            m = torch.mean(m, dim=0)
            return torch.argmax(m)
        
    def push(self, state, action, reward, next_state):
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        self.replay_memory.push((state, action, reward, next_state))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def backward(self):
        if self.replay_memory.not_enogh():
            return
        losses = []
        for state, action, reward, next_state in self.replay_memory.getBatch():
            with torch.no_grad():
                target_value = reward + self.gamma * torch.max(self.target(next_state))
            self.optimizer.zero_grad()
            q_values = self.policy(state)
            policy_output = q_values.clone()
            q_values[action] = target_value
            loss = F.mse_loss(q_values, policy_output)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()
        self.loss.append(np.mean(losses))

    def update_weight(self):
        self.target.load_state_dict(self.policy.state_dict())

class AgentCNN:
    def __init__(self, action_dim, mem_capacity = 2048, batch_size = 64, epsilon_decay = 0.99, epsilon_min = 0.01, lr = 0.5, gamma = 0.9):
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.gamma = gamma
        self.action_dim = action_dim
        self.policy = DuelCNN(84, 84, action_dim)
        self.target = DuelCNN(84, 84, action_dim)
        self.target.load_state_dict(self.policy.state_dict())
        self.replay_memory = ReplayMemory(mem_capacity, batch_size)
        self.loss = []
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            q_values = self.policy(torch.tensor(state).float())
            return torch.argmax(q_values[0])
        
    def epsilondecrease(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def push(self, state, action, reward, next_state):
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        self.replay_memory.push((state, action, reward, next_state))
        
    
    def backward(self):
        if self.replay_memory.not_enogh():
            return 0
        losses = []
        for state, action, reward, next_state in self.replay_memory.getBatch():
            with torch.no_grad():
                target_value = reward + self.gamma * torch.max(self.target(torch.tensor(next_state).float()))
            self.optimizer.zero_grad()
            q_values = self.policy(torch.tensor(state).float())
            policy_output = q_values.clone()
            q_values[0][action] = target_value
            loss = (q_values - policy_output).pow(2).mean()#F.mse_loss(q_values, policy_output)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()
        #self.loss.append(np.mean(losses))
        return np.mean(losses)

    def update_weight(self):
        self.target.load_state_dict(self.policy.state_dict())

    
"""
import cv2

model = DuelCNN(84, 84, 5)



def preProcess(image):
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To grayscale
    #frame = frame[self.crop_dim[0]:self.crop_dim[1], self.crop_dim[2]:self.crop_dim[3]]  # Cut 20 px from top
    frame = cv2.resize(frame, (84, 84))  # Resize
    frame = frame.reshape(84, 84) / 255  # Normalize
    return frame

frame = preProcess(np.float32(np.array(torch.randint(0, 255, (210, 160, 3)).tolist())))

x = torch.tensor([frame, frame, frame, frame]).unsqueeze(0)

print(model(x))
"""