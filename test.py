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
#env.env.reward_type = 'dense'
#env.compute_reward
agent = TD3(env)
noise = OUNoise(env.action_space)
batch_size = 16
rewards = []
avg_rewards = []

for episode in range(1500):
    state = env.reset()
    noise.reset()
    episode_reward = 0
    episode_list = []

    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        
        new_state, reward, done, _ = env.step(action)
        episode_list.append([state, action, reward, new_state, done])
        
        

        state = new_state
        episode_reward += reward
        if done:
            sys.stdout.write(
                "episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2),
                                                                         np.mean(rewards[-50:])))
            break
    agent.memory.push(episode_list)
    if len(agent.memory) > batch_size:
        for _ in range(20):
            agent.update(batch_size)

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-50:]))

#plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
#plt.savefig('incremental_normalizer.png')
plt.show()