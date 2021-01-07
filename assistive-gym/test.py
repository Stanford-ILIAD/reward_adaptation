import argparse
import os

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize


# workaround to unpickle olf model files
import sys
sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='ScratchItchJaco-v0',
                    help='environment to train on (default: ScratchItchJaco-v0)')
parser.add_argument('--load-dir', default='./trained_models/ppo/',
                    help='directory to save agent logs (default: ./trained_models/ppo/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
parser.add_argument('--load_epoch', type=int, default=0,
                    help='The model of which epoch to load')
parser.add_argument('--load_model', default=None, help='The model to load')
parser.add_argument('--obs_size', type=float, default=1.,
                    help='obstacle size factor')
parser.add_argument('--rew_factor', type=float, default=1.,
                    help='reward factor')
args = parser.parse_args()

from assistive_gym.envs import FeedingEnvHomotopyDownAdjust, FeedingEnvHomotopyUpAdjust
FeedingEnvHomotopyDownAdjust.obs_size = args.obs_size
FeedingEnvHomotopyUpAdjust.obs_size = args.obs_size

args.det = not args.non_det

env = make_vec_envs(args.env_name, args.seed + 1000, 1, None, None,
                    args.add_timestep, device='cpu', allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
if args.load_model is not None:
    actor_critic, ob_rms = torch.load(args.load_model) 
else:
    actor_critic, ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + "epoch_{:07d}.pt".format(args.load_epoch)))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

if render_func is not None:
    render_func('human')

obs = env.reset()

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i
reward_all = 0
while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)
    reward_all += reward
    masks.fill_(0.0 if done else 1.0)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if render_func is not None:
        render_func('human')
    if done:
      break
print(reward_all)
