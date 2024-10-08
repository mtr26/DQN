import gym

env = gym.make("CarRacing-v2", domain_randomize=True, render_mode="human")

env.action_space.shape

observation, info = env.reset()

import torch
print(torch.tensor(observation).size())

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()