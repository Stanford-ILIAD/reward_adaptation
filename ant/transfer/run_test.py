import argparse
from itertools import count

import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
import os

import ant
import pdb

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--output_dir', type=str, help='output directory')
parser.add_argument('--snapshot_path', type=str, default=None, help='snapshot path')
args = parser.parse_args()

env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

if 'PNN' in args.snapshot_path:
    policy_net = PolicyPNN(num_inputs, num_actions)
    value_net = ValuePNN(num_inputs)
else:
    policy_net = Policy(num_inputs, num_actions)
    value_net = Value(num_inputs)

if args.snapshot_path is not None:
    snapshot = torch.load(args.snapshot_path)
    policy_net.load_state_dict(snapshot['policy'])
    value_net.load_state_dict(snapshot['value'])

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

#running_state = ZFilter((num_inputs,), clip=5)
#running_reward = ZFilter((1,), demean=False, clip=10)

num_steps = 0
reward_batch = 0
num_episodes = 0

for i_episode in count(1):
    state = env.reset()

    h1 = 0
    h2 = 0
    h3 = 0

    reward_sum = 0
    for t in range(10000): # Don't infinite loop while learning
        action = select_action(state)
        action = action.data[0].numpy()
        next_state, reward, done, info = env.step(action)
        reward_sum += reward

        #next_state = running_state(next_state)

        if info['reward_homotopy'] > 0:
            h1 += 1
        elif info['reward_homotopy'] < -999:
            h2 += 1
        else:
            h3 += 1

        if args.render:
            env.render()
        if done:
            break

        state = next_state
    
    print(t, h1, h2, h3)

    num_steps += (t-1)
    num_episodes += 1
    reward_batch += reward_sum

    print('instant rew: {}, average rew: {}'.format(reward_sum, reward_batch / num_episodes))
