import numpy as np
import gym
from collections import deque
import random
import pdb
import copy

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=50000):
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
        #print(ou_state)
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

class NormalNoise(object):
    def __init__(self, action_space, decay_period=10000):
        self.decay_period = decay_period
        self.time = 0
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.time = 0
    
    def get_action(self, action, step):
        noise = np.random.normal(scale=0.1, size=self.action_dim)
        modifier = np.exp(-4*self.time/self.decay_period)
        noise = modifier*noise
        self.time += 1
        return np.clip(action + noise, self.low, self.high)

class OrnsteinUhlenbeckActionNoise():
    def __init__(self, action_space, mu=0.0, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()

    def get_action(self, action, _):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return np.clip(action + x, self.low, self.high)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.action_dim)

# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
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

class BasicNormalizer():
    def __init__(self, dims, limit):
        #self.total = np.zeros(dims)
        self.capacity = 10000
        self.dims = dims
        self.buffer = np.zeros(dims)[np.newaxis,:]

    def normalize(self, x, batch=True):
        batch_size = len(x)
        buffer_size = self.buffer.shape[0]
        addition = np.array(x)
        self.buffer = np.vstack((self.buffer, addition))
        std = np.std(self.buffer, axis=0)
        mean = np.mean(self.buffer, axis=0)
        result = (x - mean)/(std + 1e-8)

        if self.buffer.shape[0] > self.capacity:
            self.buffer = self.buffer[batch_size:]

        result = np.clip(result, -5, 5)
        return result

class Memory:
    def __init__(self, max_size, reward_func):
        self.max_size = max_size
        self.length = 0
        self.reward_func = reward_func
        # TODO: 50 should be max timestep
        self.buffer = deque(maxlen=max_size//50)
    
    def push(self, episode_list):
        states = []
        actions = []
        rewards = []
        T = len(episode_list)
        for t in range(T):
            states.append(episode_list[t][0])
            actions.append(episode_list[t][1])
            rewards.append(episode_list[t][2])

        states.append(episode_list[T-1][3])
        episode_tup = (states, actions, rewards)
        
        self.length += 1
        self.buffer.append(episode_tup)

    def sample(self, batch_size, k=5):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []

        batch = random.sample(self.buffer, batch_size)

        for episode in batch:
            states = episode[0]
            actions = episode[1]
            rewards = episode[2]
            T = len(rewards)
            # random int from 0 (inclusive) to T-k (exclusive)
            idx = np.random.randint(0, T-k)
            
            state_batch.append(states[idx])
            action_batch.append(actions[idx])
            reward_batch.append(rewards[idx])
            next_state_batch.append(states[idx+1])

            kindxs = np.random.randint(idx + 1, T, (k))
            kindxs = list(kindxs)

            for idx in kindxs:
                normal_state = copy.deepcopy(states[idx])
                action = actions[idx]
                normal_next = copy.deepcopy(states[idx+1])
                reward = 0

                new_goal = normal_next['achieved_goal']
                normal_state['desired_goal'] = new_goal
                normal_next['desired_goal'] = new_goal
                state_batch.append(normal_state)
                action_batch.append(action)
                #reward_batch.append(self.reward_func(normal_next['achieved_goal'], new_goal))
                reward_batch.append(reward)
                next_state_batch.append(normal_next)
        
        reward_batch = np.array(reward_batch)
        reward_batch = reward_batch.reshape(-1, 1)

        return state_batch, action_batch, reward_batch, next_state_batch

    def __len__(self):
        return self.length