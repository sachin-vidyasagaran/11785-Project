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
from ddpgwher import *


if __name__ == "__main__":
    env = gym.make('FetchReach-v1')
    env = NormalizedEnv(env)

    agentwDDPGHER = DDPGagentwithHER(env, [256])
    noise = OUNoise(env.action_space)
    # with HER
    batch_size = 128
    rewards_her_new = []
    avg_rewards_her_new = []
    d_goal = np.array([1, 0, 0])
    temp_buffer = []
    sample_buffer=[]
    batch_buffer = []
    for episode in range(100):
        episode_memory = Memorywithgoal(batch_size)
        state = env.reset()
        print('state', state)
        noise.reset()
        episode_reward = 0
        # if len(agentwDDPGHER.memory) > batch_size:
        # for step in range(500):
        #         #         # states, actions, rewards, next_states, _, goal = agentwDDPGHER.memory.sample(batch_size)
        #         #         # agentwDDPGHER.memory
        #         #     action = agentwDDPGHER.get_action(state['observation'], d_goal)
        #         #     action = noise.get_action(action, step)
        #         #     new_state, reward, done, _ = env.step(action)
        #         #     sample_buffer.append((state,action,new_state))
        #         #     state = new_state
        for step in range(50):

            action = agentwDDPGHER.get_action(state['observation'], d_goal)
            action = noise.get_action(action, step)
            new_state, reward, done, _ = env.step(action)

            agentwDDPGHER.memory.push(state, action, reward, new_state, done, d_goal)
            temp_buffer.append((state, action, reward, new_state, done))
            a_goal = new_state
            #
            #     agentwDDPGHER.update(batch_size)

            state = new_state
            episode_reward += reward
            length = len(temp_buffer)

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
            episode_memory.push(state, action, preward, new_state, done, a_goal)
            # agentwDDPGHER.memory.push(state, action, preward, new_state, done, a_goal)
        #
        if len(agentwDDPGHER.memory) > batch_size:
            # temp_buffer = []
            for step in range(50):
                agentwDDPGHER.updateUsingHer(batch_size,episode_memory, )
        temp_buffer = []
        rewards_her_new.append(episode_reward)
        avg_rewards_her_new.append(np.mean(rewards_her_new[-10:]))
