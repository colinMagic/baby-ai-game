#!/usr/bin/env python3

import time
import random
import numpy as np
import operator
from functools import reduce

import gym
import gym_minigrid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

class Model(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()

        self.num_inputs = reduce(operator.mul, input_size, 1)
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.num_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

        self.optimizer = optim.SGD(
            self.parameters(),
            lr=0.0005,
            momentum=0.4
        )

    def forward(self, obs):
        image = obs['image']

        # FIXME: what's the unsqueeze for?
        image = Variable(torch.from_numpy(image).float().unsqueeze(0))

        # Reshape the input so that it is one-dimensional
        image = image.view(-1, self.num_inputs)

        x = F.relu(self.fc1(image))
        x = F.relu(self.fc2(x))
        action_scores = self.fc3(x)
        action_probs = F.softmax(action_scores, dim=1)

        return action_probs

    def select_action(self, obs, sample=True):
        action_probs = self.forward(obs)
        if sample:
            dist = Categorical(action_probs)
            action = dist.sample()
            #log_prob = dist.log_prob(action)
        else:
            _, action = torch.max(action_probs, 1)

        return action.data[0]

    def action_log_prob(self, obs, action):
        action_probs = self.forward(obs)

        action = Variable(torch.LongTensor(1).fill_(action))

        dist = Categorical(action_probs)
        log_prob = dist.log_prob(action)

        return log_prob

class Rollout:
    def __init__(self, seed=0, action=None, obs=None):
        if action is None:
            action = []
        if obs is None:
            obs = []
        self.seed = seed
        self.obs = obs
        self.action = action
        self.reward = [] if not action else [0] * (len(action) - 1) + [1000 - len(action)]
        self.length = len(action)
        self.total_reward = 0 if not action else 1000 - len(action)

    def append(self, obs, action, reward):
        self.obs.append(obs)
        self.action.append(int(action))
        self.reward.append(reward)
        self.length += 1
        self.total_reward += reward

    def __repr__(self):
        return str({'total_reward': self.total_reward,
                'all': {i: (self.obs[i]['image'].max(axis=0).max(axis=0), self.action[i], self.reward[i]) for i in range(self.length)}
                })

def random_rollout(env, seed):
    env.seed(seed)
    obs = env.reset()

    while True:
        action = random.randint(0, env.action_space.n - 1)
        newObs, reward, done, info = env.step(action)

        rollout.append(obs, action, reward)
        obs = newObs

        if done:
            break

    return rollout

def equiv_rollout(env, r1):
    num_trials = 0

    while True:
        r2 = random_rollout(env, r1.seed)
        num_trials += 1

        print(num_trials)

        if len(r2.obs) <= len(r1.obs):
            return r2

def run_model(model, env, seed, eps=0, sample=True):
    env.seed(seed)
    obs = env.reset()
    rollout = Rollout(seed)

    while True:
        if not isinstance(obs, dict):
            obs = { 'image': obs, 'mission': '' }

        if random.random() < eps:
            action = random.randint(0, env.action_space.n - 1)
        else:
            action = model.select_action(obs, sample=sample)

        newObs, reward, done, info = env.step(action)

        rollout.append(obs, action, reward)
        obs = newObs

        if done:
            break

    return rollout

def cross_entropy(yHat, y):
    if yHat == 1:
      return -log(y)
    else:
      return -log(1 - y)

def train_model(model, rollout):
    losses = []

    for step in range(0, len(rollout.obs)):
        rollout_action = rollout.action[step]
        log_prob = model.action_log_prob(rollout.obs[step], rollout_action)
        losses.append(-log_prob)

    model.optimizer.zero_grad()
    loss = torch.cat(losses).sum()

    loss.backward()
    model.optimizer.step()

    return loss.cpu().data[0]

def eval_model(model, env, num_evals=64):
    sum_reward = 0
    obs = env.reset()

    for n in range(0, num_evals):
        while True:
            action = model.select_action(obs)
            obs, reward, done, info = env.step(action)
            sum_reward += reward
            if done:
                break

    return sum_reward / num_evals
