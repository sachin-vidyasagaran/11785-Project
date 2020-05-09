import sys
import gym
import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from td3 import TD3
from td4 import TD4
from utils import *
import random

env = gym.make("FetchReach-v1")
#env.env.reward_type = 'dense'  # WARNING! HER implemented currently only for sparse rewards. Dense will break it!
agent = DDPGagent(env)
# agent = TD3(env) # Uncomment for TD3
# agent = TD4(env) # Uncomment for Extended TD3
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []

for episode in range(10000):
    state = env.reset()
    noise.reset()
    episode_reward = 0
    agent.memory.clear_trajectory()

    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, new_state, done)
        # env.render()
        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = new_state
        episode_reward += reward

        if done:
            agent.memory.HER_future(state, step+1, env.compute_reward) # Future Strategy
            # agent.memory.HER(state, step+1, env.compute_reward) # Final Strategy
            sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-50:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-50:]))
    if (episode > 1500):
        break
    if (episode%100 == 0):
        agent.save_models()

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
