import gym

import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import models_goal
from utils import *
from ddpgwher import *
from models_goal import *

if __name__ == "__main__":
    env = gym.make('FetchReach-v1')
    env = NormalizedEnv(env)

    agentwDDPGHER = DDPGagentwithHER(env, [1024, 512, 256])
    noise = OUNoise(env.action_space)
    # with HER
    batch_size = 128
    rewards_her_new = []
    avg_rewards_her_new = []
    # d_goal = np.array([env.goal_position, env.goal_velocity])
    temp_buffer = []

    for episode in range(10000):
        episode_memory = Memorywithgoal(batch_size)
        state = env.reset()
        d_goal = state['desired_goal']
        # print('state', state)
        noise.reset()
        episode_reward = 0

        for step in range(50):
            action = agentwDDPGHER.get_action(state['observation'], d_goal)
            action = noise.get_action(action, step)
            new_state, reward, done, _ = env.step(action)

            agentwDDPGHER.memory.push(state['observation'], action, reward, new_state['observation'], done, d_goal)
            temp_buffer.append((state['observation'], action, reward, new_state['observation'], done))
            a_goal = new_state['achieved_goal']
            #             if len(agentwDDPGHER.memory) > batch_size:
            #                 #             length=len(temp_buffer)

            #                 agentwDDPGHER.update(batch_size)

            state = new_state
            episode_reward += reward

            

                # temp_buffer = []

            if done:
                sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode,
                                                                                          np.round(episode_reward,
                                                                                                   decimals=2), np.mean(
                        rewards_her_new[-10:])))
                break
        length = len(temp_buffer)
        for idx, (mems) in enumerate(temp_buffer):

            state, action, reward, new_state, done = mems
            preward = reward
            if idx == length - 1:
                #                     print('yes')
                preward = 5
            #                 print('r',reward)
            episode_memory.push(state, action, preward, new_state, done, a_goal)
        if len(agentwDDPGHER.memory) > batch_size:
            # temp_buffer = []
            for step in range(50):
                agentwDDPGHER.updateUsingHer(batch_size, episode_memory, length)
        temp_buffer = []

    rewards_her_new.append(episode_reward)
    avg_rewards_her_new.append(np.mean(rewards_her_new[-10:]))
plt.plot(avg_rewards)
# plt.plot(rewards_her)
plt.plot(avg_rewards_her_new)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
