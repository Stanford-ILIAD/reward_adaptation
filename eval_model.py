import os
import collections
import pickle
import shutil
import csv
import gym
import numpy as np
import time
from stable_baselines import DQN
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
import wandb
from tensorflow import flags
import minigrid.gym_minigrid
import utils

def load_env(env, num_envs=1):
    env_fns = num_envs * [lambda: gym.make(env)]
    eval_env = VecNormalize(DummyVecEnv(env_fns), training=False, norm_reward=False)
    #env = VecNormalize(SubprocVecEnv(env_fns))
    #env = VecNormalize(env_fns)
    return eval_env

def load_model(model_dir):
    policy = MlpPolicy
    model = DQN.load(model_dir)
    return model

def evaluate(model, eval_env):
    """
    Evaluates model on 10 episodes of driving task.
    Returns mean episode reward and standard deviation.
    """
    total_rets = []
    for e in range(1):
        rets = 0.0
        obs = eval_env.reset()
        state, done = None, False
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            next_obs, ret, done, _info = eval_env.step(action)
            eval_env.render()
            if not done:
                rets += ret
            obs = next_obs
            time.sleep(.1)
        total_rets.append(rets)
    return np.mean(total_rets), np.std(total_rets), total_rets

if __name__ == "__main__":
    gwv0_unnorm = ("gridworld", "no_norm", "best_model_3000_3.888888888888889.pkl")
    gwv0_norm = ("gridworld", "norm", "best_model_42000_0.4861111111111111.pkl")
    model = gwv0_unnorm
    model_dir = os.path.join(model[0], model[1], model[2])
    eval_env = load_env("Gridworld-v0")

    model = load_model(model_dir)
    sum_reward = 0
    num_episode = 200
    for ne in range(num_episode):
        sum_reward += evaluate(model, eval_env)
        print("running mean: ", sum_reward/(ne+1))

    print("mean ret: ", sum_reward/num_episode)
