import os
import collections
import pickle
import shutil
import csv
import gym
import numpy as np
import time
from stable_baselines import DQN, PPO2, HER, DDPG
from stable_baselines.her.utils import HERGoalEnvWrapper
from stable_baselines.common.policies import MlpPolicy
import wandb
from tensorflow import flags
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import driving.driving_envs
import fetch.fetch_envs
import ipdb
from tensorflow import flags
import stable_baselines


FLAGS = flags.FLAGS
flags.DEFINE_string("env", "fetch", "environment")

def load_env(env, num_envs=1):
    eval_env = gym.make(env)
    return eval_env


def load_model(model_info, model_type="PPO", baseline=None, pkl_file=None):
    model_dir = os.path.join(model_info[0], model_info[1], model_info[2])
    if model_type == "PPO":
        if baseline == 'L2SP':
            from baselines.L2SP.model import PPO2L2SP
            import baselines.L2SP.utils as L2SP_utils
            data, params = L2SP_utils.load_from_file(model_dir)
            model = PPO2L2SP.load(model_dir, original_params=params)
        elif baseline == 'PNN':
            from baselines.PNN.utils import looseload, resave_params_for_PPN
            output_dir = os.path.join("output/updated_gridworld_continuous_PNN", 'resave', model_info[2])
            resave_params_for_PPN(model_dir, output_dir)
            model = looseload(PPO2, output_dir)
        elif baseline == 'BSS':
            from baselines.BSS.utils import resave_params_for_BSS
            from baselines.BSS.model import PPO2BSS
            output_dir = os.path.join("output/updated_gridworld_continuous_BSS", 'resave', model_info[2])
            resave_params_for_BSS(model_dir, output_dir)
            model = PPO2BSS.load(output_dir, bss_coef=0.001, l2_coef=0.0005)
        else:
            model = PPO2.load(model_dir)

    elif model_type == "HER":
        if baseline == 'L2SP':
            from baselines_fetch.L2SP.model import HER2L2SP
            import baselines_fetch.L2SP.utils as L2SP_utils
            data, params = L2SP_utils.load_from_file(model_dir)
            model = HER2L2SP.load(model_dir, original_params=params)
        elif baseline == "PNN":
            from baselines_fetch.PNN.model import HER2PNN
            from baselines_fetch.PNN.utils import resave_params_for_PNN
            output_dir = os.path.join("output/fetch_PNN", 'resave', model_info[2])
            resave_params_for_PNN(model_dir, output_dir)
            model = HER2PNN.load(output_dir)
        elif baseline == "BSS":
            pass
        else:
            model = HER.load(model_dir)
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
        if isinstance(eval_env, stable_baselines.her.utils.HERGoalEnvWrapper):  # fetch reach env, saving xyz of end effector
            state_history.append(obs[:3])
        else:  # eval env is driving environment
            state_history.append(obs[:2])
        state, ever_done = None, False
        while not ever_done:
            if render: eval_env.render()
            nsteps += 1
            action, state = model.predict(obs, state=state, deterministic=True)
            next_obs, ret, done, _info = eval_env.step(action, verbose=render)
            if render: eval_env.render()
            if not ever_done:
                rets += ret
            obs = next_obs
            if isinstance(eval_env, stable_baselines.her.utils.HERGoalEnvWrapper):
                state_history.append(obs[:3])
            else:  # eval env is driving environment
                state_history.append(obs[:2])
            if render: time.sleep(.1)
            ever_done = done
        if render: eval_env.render()
        total_rets.append(rets)
    return np.mean(total_rets), np.std(total_rets), total_rets, np.array(state_history)

def save_traj(model, state_history):
    state_history = list(state_history)
    with open("output/fetch/single_trajs/{}.csv".format(model[1]), "w") as f:
        writer = csv.writer(f)
        writer.writerow(state_history)

if __name__ == "__main__":
    if FLAGS.env == "nav1":
        from output.updated_gridworld_continuous.policies import *
        model_info = B5R_B5L
        eval_env = load_env("Continuous-v0", "PPO")
        model = load_model(model_info, "PPO", baseline=None)
    elif FLAGS.env == 'fetch':
        from output.fetch2.policies import *
        model_info = BR_BL
        eval_env = HERGoalEnvWrapper(load_env("Fetch-v0"))
        model = load_model(model_info, "HER", baseline=None)
    save = False
    sum_reward = 0.0
    num_episode = 10
    for ne in range(num_episode):
        mean_ret, std_ret, total_ret, state_history = evaluate(model, eval_env, render=True)
        save_traj(model_info, state_history)
        sum_reward += mean_ret
        print("\nrunning mean: ", sum_reward / (ne + 1))

    print("mean ret: ", sum_reward / num_episode)
