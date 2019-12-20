import os
import collections
import pickle
import shutil
import csv
import gym
import numpy as np
import time
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
import wandb
from tensorflow import flags
import driving_envs


if __name__ == "__main__":
    env = gym.make("Navigation-v0")
    obs = env.reset()
    for _ in range(100):
        env.step([1, 1])
        env.render()
