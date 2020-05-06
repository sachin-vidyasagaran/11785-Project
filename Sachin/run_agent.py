import torch
import os
from models_td3 import *
from td4 import *

def main():
    env = gym.make("FetchReach-v1")
    agent = TD4(env)
    agent.load_models()
    agent.statenorm.counter, agent.statenorm.mean, agent.statenorm.varhelper, \
    agent.statenorm.var, agent.statenorm.dims, agent.statenorm.length = torch.load('saved_models/saved_normalizer')
    env.reset()

    demo_len = 300
    for i in range(demo_len):
        state = env.reset()
        for t in range(env._max_episode_steps):
            env.render()
            with torch.no_grad():
                pi = agent.get_action(state)
            action = pi#.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))

if __name__ == '__main__':
    main()