import numpy as np
import gym
from collections import deque
import random
import pdb

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
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
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
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