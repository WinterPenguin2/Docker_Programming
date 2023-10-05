import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# 환경 설정
env = gym.make("Pendulum-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_high = env.action_space.high
action_low = env.action_space.low

# 신경망 모델 정의
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DDPG 에이전트 설정
actor = Actor()
critic = Critic()
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

# 하이퍼파라미터 설정
gamma = 0.99
tau = 0.001
batch_size = 64
buffer_size = 10000
buffer = []

# 타겟 네트워크 설정
target_actor = Actor()
target_critic = Critic()
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

# 랜덤한 초기화 함수
def initialize_random_buffer():
    while len(buffer) < buffer_size:
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            state = next_state

# 타겟 네트워크 업데이트 함수
def update_target_networks():
    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# DDPG 알고리즘 메인 루프
initialize_random_buffer()
for episode in range(1000):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = actor(torch.FloatTensor(state)).detach().numpy()
        action += np.random.normal(0, 0.1, action_dim)
        action = np.clip(action, action_low, action_high)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        buffer.append((state, action, reward, next_state, done))
        
        if len(buffer) > batch_size:
            minibatch = random.sample(buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            target_actions = target_actor(next_states).detach()
            target_values = target_critic(next_states, target_actions).detach()
            target_q_values = rewards + gamma * target_values * (1 - dones)
            
            critic_optimizer.zero_grad()
            q_values = critic(states, actions)
            critic_loss = nn.MSELoss()(q_values, target_q_values)
            critic_loss.backward()
            critic_optimizer.step()

            actor_optimizer.zero_grad()
            actor_loss = -critic(states, actor(states)).mean()
            actor_loss.backward()
            actor_optimizer.step()

            update_target_networks()

        state = next_state

    print(f"Episode: {episode}, Total Reward: {total_reward}")

env.close()
