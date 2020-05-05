import numpy as np
import gym
from collections import deque
import random
import pdb
import copy

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.3, max_sigma=0.3, min_sigma=0.1, decay_period=50):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)


class Normalizer():
    def __init__(self, dims, limit):
        #self.total = np.zeros(dims)
        self.counter = 0
        self.mean = np.zeros(dims - limit)
        self.varhelper = np.ones(dims - limit)
        self.var = np.ones(dims - limit)
        self.dims = dims
        self.length = dims - limit

    def normalize(self, x, batch=True):
        batch_size = len(x)

        if batch:
            res = np.zeros((batch_size, self.length))
            for b in range(batch_size):
                #self.total += x
                self.counter += 1

                res[b] = (x[b][:self.length] - self.mean)/self.var

                #http://datagenetics.com/blog/november22017/index.html
                #recalculate mean
                oldmean = self.mean.copy()
                self.mean += (x[b][:self.length] - self.mean)/self.counter
                #recalculate variance
                self.varhelper += (x[b][:self.length] - oldmean)*(x[b][:self.length] - self.mean)
                self.var = np.sqrt(self.varhelper/self.counter)
            result = np.zeros((batch_size, self.dims))
            for b in range(batch_size):
                result[b] = np.hstack((res[b], x[b][self.length:]))
        else:
            res = np.zeros(self.length)
            self.counter += 1

            res = (x[:self.length] - self.mean) / self.var

            # recalculate mean
            oldmean = self.mean.copy()
            self.mean += (x[:self.length] - self.mean) / self.counter
            # recalculate variance
            self.varhelper += (x[:self.length] - oldmean)*(x[:self.length] - self.mean)
            self.var = np.sqrt(self.varhelper / self.counter)
            result = np.hstack((res, x[self.length:]))
        result = np.clip(result, -5, 5)
        return result

class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.traj = []

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.traj.append(experience)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

    ''' HER Functions '''

    def clear_trajectory(self):
        self.traj = []

    def HER(self, final_state, final_timestep, reward_function):
        subs_goal = final_state['achieved_goal']
        # print("SUBS GOAL:\n",subs_goal)
        assert(len(self.traj) == final_timestep)

        for t in range(final_timestep):

            # print("-"*50, "\nTimestep: ", t)
            # print(self.traj[t])

            state, action, reward, next_state, done = copy.copy(self.traj[t])  # Unpack tuple

            her_state = copy.copy(state)
            her_state['desired_goal'] = subs_goal

            # print(her_state['desired_goal'])
            # print(state['desired_goal'])

            her_next_state = copy.copy(next_state)
            her_next_state['desired_goal'] = subs_goal

            # TODO:
            # When your next state's achieved goal is the substitute goal.
            # Call the env.compute reward with the above parameters and set that as the reward

            # her_reward = 0. if done else -1. # Sparse Rewards
            her_reward = reward_function(her_next_state['achieved_goal'], subs_goal, None)
            # if her_reward == 0.:
            #     print("Reward 0 at timestep:", t)

            # print(state['desired_goal'])
            # print(next_state['desired_goal'])
            hindsight_experience = (her_state, action, np.array([her_reward]), her_next_state, done)

            # print("__________UPDATED:")
            # print(hindsight_experience)

            self.buffer.append(hindsight_experience)



    def HER_future(self, final_state, final_timestep, reward_function):

        k = 6
        assert(len(self.traj) == final_timestep)

        for t in range(final_timestep-k):

            # print("-"*50, "\nTimestep: ", t)
            # print(self.traj[t])

            t_subs = np.random.randint(low=t+1, high=final_timestep, size=k)  # Sample k timesteps corresponding to the substitute goals
            subs_goals = [self.traj[t_sub][3]['achieved_goal'] for t_sub in t_subs]

            for i in range(k):    # Iterate over the k substitute goals

                t_sub = t_subs[i]
                subs_goal = subs_goals[i]

                for t_s in range(t, t_sub): # Iterate from timestep t to the timestep of each substitute goal

                    state, action, reward, next_state, done = copy.copy(self.traj[t_s])  # Unpack tuple

                    her_state = copy.copy(state)
                    her_state['desired_goal'] = subs_goal

                    # print(her_state['desired_goal'])
                    # print(state['desired_goal'])

                    her_next_state = copy.copy(next_state)
                    her_next_state['desired_goal'] = subs_goal

                    her_reward = reward_function(her_next_state['achieved_goal'], subs_goal, None)
                    

                    # print(state['desired_goal'])
                    # print(next_state['desired_goal'])
                    hindsight_experience = (her_state, action, np.array([her_reward]), her_next_state, done)

                    # print("__________UPDATED:")
                    # print(hindsight_experience)

                    self.buffer.append(hindsight_experience)