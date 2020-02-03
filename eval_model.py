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

def load_env(env, num_envs=1):
    #env_fns = num_envs * [lambda: gym.make(env)]
    #eval_env = VecNormalize(DummyVecEnv(env_fns), training=False, norm_reward=False)
    eval_env = gym.make(env)
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
    nsteps = 0
    for e in range(1):
        rets = 0.0
        obs = eval_env.reset()
        state, ever_done = None, False
        while not ever_done:
            nsteps +=1
            action, state = model.predict(obs, state=state, deterministic=True)
            next_obs, ret, done, _info = eval_env.step(action, verbose=False)
            #print("ret: ", ret)
            #eval_env.render()
            if not ever_done:
                rets += ret
            #print("rets: ", rets)
            obs = next_obs
            #time.sleep(.1)
            ever_done = done
        total_rets.append(rets)
        #print("total mean ep return: ", np.mean(total_rets), total_rets)
        #print("nsteps: ", nsteps)
    return np.mean(total_rets), np.std(total_rets), total_rets

if __name__ == "__main__":
    gw1 = ("gridworld", "v0", "best_model_5000_101.0.pkl")
    model = gw1
    model_dir = os.path.join(model[0], model[1], model[2])
    eval_env = load_env("Gridworld-v0")

    model = load_model(model_dir)
    sum_reward = 0.0
    num_episode = 200
    for ne in range(num_episode):
        mean_ret, std_ret, total_ret = evaluate(model, eval_env)
        sum_reward += mean_ret
        print("running mean: ", sum_reward/(ne+1))

    print("mean ret: ", sum_reward/num_episode)
