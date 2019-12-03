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


def evaluate_debug(model_dir, eval_dir=None, num_envs=1):
    """
    Evaluates model on one episode of driving task. Returns mean episode reward.
    """
    env_fns = num_envs * [lambda: gym.make("Merging-v0")]
    eval_env = VecNormalize(DummyVecEnv(env_fns), training=False, norm_reward=False)
    env = VecNormalize(SubprocVecEnv(env_fns))
    policy = MlpPolicy
    # model = PPO2(policy, env, verbose=1)
    model = PPO2.load(model_dir)

    rets = 0.0
    obs = eval_env.reset()
    state, ever_done = None, False
    while not ever_done:
        action, state = model.predict(obs, state=state, deterministic=True)
        # print("\naction: ", action)
        next_obs, rewards, done, _info = eval_env.step(action)
        print("rewards: ", rewards)
        # if not is_save: eval_env.render()
        eval_env.render()
        if not ever_done:
            rets += rewards
        ever_done = np.logical_or(ever_done, done)
        obs = next_obs
        # if not is_save: time.sleep(.1)
        time.sleep(.1)
    eval_env.close()
    return rets


def evaluate(model_dir, num_envs=1):
    """
    Evaluates model on one episode of driving task. Returns mean episode reward.
    """

    env_fns = num_envs * [lambda: gym.make("Merging-v0")]
    eval_env = VecNormalize(DummyVecEnv(env_fns), training=False, norm_reward=False)
    env = VecNormalize(SubprocVecEnv(env_fns))
    policy = MlpPolicy
    model = PPO2(policy, env, verbose=1)
    PPO2.load(model_dir)

    num_iters = 3
    for _ in range(num_iters):
        obs = eval_env.reset()
        rets = np.zeros(num_envs)
        state, dones = None, [False for _ in range(num_envs)]
        ever_done = np.zeros((num_envs,), dtype=np.bool)
        task_data = collections.defaultdict(list)  # Maps env_idx -> (state, action, reward, done) tuples
        while not np.all(ever_done):
            true_states = [
                inner_env.world.state for inner_env in eval_env.venv.envs
            ]
            action, state = model.predict(obs, state=state, mask=dones, deterministic=True)
            next_obs, rewards, dones, _info = eval_env.step(action)
            eval_env.render()
            for env_idx, data in enumerate(zip(true_states, action, rewards, dones)):
                if not ever_done[env_idx]:
                    task_data[env_idx].append(data)
                    rets[env_idx] += rewards[env_idx]
            ever_done = np.logical_or(ever_done, dones)
            obs = next_obs
            time.sleep(.1)
    return np.mean(rets / num_iters)


if __name__ == "__main__":

    safe0 = ("safe0", "eval559best_model_559_[710.741].pkl")
    eff100 = ("eff100", "eval119best_model_119_[58557.055].pkl")
    eff = ("eff", "eval489best_model_489_[-29.615425].pkl")
    curr02 = ("0.2_curr", "eval49best_model_49_[235.07114].pkl")
    curr2 = ("2.0_curr", "eval19best_model_19_[-56.795643].pkl")
    rand_start02 = ("0.2rand_start", "eval99best_model_99_[-40.224064].pkl")
    counterbalanced = ("counterbalanced", "eval9best_model_9_[-28.716702].pkl")
    model = safe0
    model_dir = os.path.join("reward_curriculum_expts", model[0], model[1])
    for _ in range(10):
        mean_ret = evaluate_debug(model_dir)
        print("mean ret: ", mean_ret)
