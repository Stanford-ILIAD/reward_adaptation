import os
import collections
import pickle
import shutil
import csv
import gym
import numpy as np
import time
from stable_baselines import DQN, PPO2
import wandb
from tensorflow import flags
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



def load_env(env, num_envs=1):
    # env_fns = num_envs * [lambda: gym.make(env)]
    # eval_env = VecNormalize(DummyVecEnv(env_fns), training=False, norm_reward=False)
    eval_env = gym.make(env)
    # env = VecNormalize(SubprocVecEnv(env_fns))
    # env = VecNormalize(env_fns)
    return eval_env


def load_model(model_dir, model_type="PPO"):
    #policy = MlpPolicy
    if model_type == "PPO":
        model = PPO2.load(model_dir)
    elif model_type == "DQN":
        model = DQN.load(model_dir)
    return model


def evaluate(model, eval_env, render=False):
    """
    Evaluates model on 10 episodes of driving task.
    Returns mean episode reward and standard deviation.
    """
    total_rets = []
    nsteps = 0
    state_history = []
    for e in range(1):
        rets = 0.0
        obs = eval_env.reset()
        state_history.append(obs[:2])
        state, ever_done = None, False
        while not ever_done:
            nsteps += 1
            action, state = model.predict(obs, state=state, deterministic=True)
            next_obs, ret, done, _info = eval_env.step(action, verbose=render)
            # print("ret: ", ret)
            if render: eval_env.render()
            if not ever_done:
                rets += ret
            # print("rets: ", rets)
            obs = next_obs
            state_history.append(obs[:2])
            if render: time.sleep(.1)
            ever_done = done
        total_rets.append(rets)
        print("total mean ep return: ", np.mean(total_rets), total_rets)
        print("nsteps: ", nsteps)
    return np.mean(total_rets), np.std(total_rets), total_rets, np.array(state_history)

def save_traj(model, state_history):
    state_history = list(state_history)
    with open("output/gridworld_continuous/single_trajs/{}.csv".format(model[1]), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(state_history)

if __name__ == "__main__":
    #from gridworld_policies.policies import *
    from output.gridworld_continuous.policies import *

    model_info = B6R
    model_dir = os.path.join(model_info[0], model_info[1], model_info[2])
    eval_env = load_env("Continuous-v0", "PPO")
    save = True

    model = load_model(model_dir)
    sum_reward = 0.0
    num_episode = 200
    for ne in range(num_episode):
        mean_ret, std_ret, total_ret, state_history = evaluate(model, eval_env, render=True)
        save_traj(model_info, state_history)
        sum_reward += mean_ret
        print("\nrunning mean: ", sum_reward / (ne + 1))
        break

    print("mean ret: ", sum_reward / num_episode)
