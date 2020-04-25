
import gym


from models import Actor,Critic

import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from models import *

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import models_goal
from utils import *


if __name__ == "__main__":
    env = NormalizedEnv(gym.make("Pendulum-v0"))

    agentwDDPGHER = DDPGagentwithHER(env, [256])
    noise = OUNoise(env.action_space)
    # with HER
    batch_size = 128
    rewards_her_new = []
    avg_rewards_her_new = []
    d_goal = np.array([1, 0, 0])
    temp_buffer = []

    for episode in range(100):
        state = env.reset()
        print('state', state)
        noise.reset()
        episode_reward = 0

        for step in range(500):
            action = agentwDDPGHER.get_action(state['observation'], d_goal)
            action = noise.get_action(action, step)
            new_state, reward, done, _ = env.step(action)

            agentwDDPGHER.memory.push(state, action, reward, new_state, done, d_goal)
            temp_buffer.append((state, action, reward, new_state, done))
            a_goal = new_state
            if len(agentwDDPGHER.memory) > batch_size:
                length = len(temp_buffer)

                agentwDDPGHER.update(batch_size)

            state = new_state
            episode_reward += reward

            if done:
                #             print('rdone',reward)
                #             print('sdone',new_state)
                for idx, (mems) in enumerate(temp_buffer):

                    state, action, reward, new_state, done = mems
                    preward = reward
                    if idx == length - 1:
                        #                     print('yes')
                        preward = 5
                    #                 print('r',reward)

                    agentwDDPGHER.memory.push(state, action, preward, new_state, done, a_goal)
                temp_buffer = []
                sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode,
                                                                                          np.round(episode_reward,
                                                                                                   decimals=2), np.mean(
                        rewards_her_new[-10:])))
                break
        for idx, (mems) in enumerate(temp_buffer):

            state, action, reward, new_state, done = mems
            preward = reward
            if idx == length - 1:
                #                     print('yes')
                preward = 5
            #                 print('r',reward)

            agentwDDPGHER.memory.push(state, action, preward, new_state, done, a_goal)
        temp_buffer = []

        rewards_her_new.append(episode_reward)
        avg_rewards_her_new.append(np.mean(rewards_her_new[-10:]))





