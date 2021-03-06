from ddpg import Agent
import gym
import numpy as np
from utils import plotLearning

env = gym.make('FetchReach-v1')
# env = gym.make('HalfCheetah-v2')
# env = gym.make('LunarLanderContinuous-v2')
env_params = env.reset()
obs_shape = env_params['observation'].shape[0]
d_goal_shape =  env_params['desired_goal'].shape[0]# Remove 'observation' indexing for envs with no dict
total_actions = act = env.action_space.sample().shape[0]
total_inputs = obs_shape + d_goal_shape
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[total_inputs], tau=0.001, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=total_actions)

#agent.load_models()
np.random.seed(0)

score_history = []
for i in range(100000):
    env_params = env.reset()
    obs = env_params['observation'] # Remove 'observation' indexing for envs with no dict
    d_goal = env_params['desired_goal']
    net_input = np.hstack((obs,d_goal))
    # print(obs.shape)
    done = False
    score = 0
    while not done:
        act = agent.choose_action(net_input)
        # print(act)
        new_state, reward, done, info = env.step(act)
        new_state = np.hstack((new_state['observation'],new_state['desired_goal'])) ## Remove 'observation' indexing for envs with no dict
        agent.remember(net_input, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        net_input = new_state
        #env.render()
    score_history.append(score)

    if i % 25 == 0:
        agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

filename = 'LunarLander-alpha000025-beta00025-400-300.png'
plotLearning(score_history, filename, window=100)
