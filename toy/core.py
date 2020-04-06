import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import random
import ipdb


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    def init_weights(m):
        print("initializing weights!!")
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1], bias=False), act()]
    net = nn.Sequential(*layers)
    net.apply(init_weights)
    return net


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        print("\nmu: ", mu, "obs: ", obs)
        #print([x for x in self.mu_net.named_parameters()])
        std = torch.exp(self.log_std)
        std = torch.Tensor([0.01])
        print("std: ", std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, exploration,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        self.exploration = exploration
        self.action_space = action_space

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, is_exp=True):
        # normalize obs
        mean_obs = 25 # max inp: 50, min inp: 0
        obs = (obs - mean_obs)/mean_obs
        #for o in range(len(obs)):
        #    if obs[o] == 0.0:
        #        obs[o]+= 1e-4
        assert (obs <= 1.5).all() and (obs >=-1.5).all()

        # scale obs in range of action space
        #print("old obs: ", obs)
        obs *= torch.Tensor(self.action_space.high)
        #print("new obs: ", obs)
        assert (obs <= 0.08).all() and (obs >= -0.08).all()

        if is_exp and random.random() < self.exploration:
            with torch.no_grad():
                pi = self.pi._distribution(obs)
                #a = pi.sample()
                a = torch.Tensor(self.action_space.sample())
                logp_a = self.pi._log_prob_from_distribution(pi, a)
                v = self.v(obs)

        else:
            with torch.no_grad():
                pi = self.pi._distribution(obs)
                a = pi.sample()
                logp_a = self.pi._log_prob_from_distribution(pi, a)
                #print("sampled a: ", a, torch.exp(logp_a))
                v = self.v(obs)

        #with torch.no_grad():
        #    pi = self.pi._distribution(obs)
        #    a = pi.sample()
        #    logp_a = self.pi._log_prob_from_distribution(pi, a)
        #    v = self.v(obs)

        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs, is_exp=False)[0]
