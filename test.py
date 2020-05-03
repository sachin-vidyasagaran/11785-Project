import sys
import gym
import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from td3 import TD3
from utils import *
import random

env = gym.make("FetchReach-v1")
# env.env.reward_type = 'dense'
agent = TD3(env)
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []

for episode in range(10000):
    state = env.reset()
    noise.reset()
    episode_reward = 0

    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, new_state, done)

        if len(agent.memory) > 500:
            agent.update(batch_size)

        state = new_state
        episode_reward += reward
        if done:
            sys.stdout.write(
                "episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2),
                                                                         np.mean(rewards[-50:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-50:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()