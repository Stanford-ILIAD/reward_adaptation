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

def load_env(num_envs=1):
    env_fns = num_envs * [lambda: gym.make("Navigation-v41")]
    eval_env = VecNormalize(DummyVecEnv(env_fns), training=False, norm_reward=False)
    #env = VecNormalize(SubprocVecEnv(env_fns))
    #env = VecNormalize(env_fns)
    return eval_env

def load_model(model_dir):
    policy = MlpPolicy
    # model = PPO2(policy, env, verbose=1)
    model = PPO2.load(model_dir)
    return model


def evaluate_debug(model, eval_env, eval_dir=None):
    """
    Evaluates model on one episode of driving task. Returns mean episode reward.
    """

    rets = 0.0
    obs = eval_env.reset()
    state, ever_done = None, False
    while not ever_done:
        action, state = model.predict(obs, state=state, deterministic=True)
        # print("\naction: ", action)
        next_obs, rewards, done, _info = eval_env.step(action)
        #print("rewards: ", rewards)
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

    env_fns = num_envs * [lambda: gym.make("Navigation-v0")]
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
  scenario = 'navigation'
  if scenario == 'navigation':
    navigation_down_5 = ("navigation_easy_central_sudden_death_down", "best_model_1487360_[-300581.84].pkl")
    navigation_up_5 = ("navigation_easy_central_sudden_death_up", "best_model_1487360_[-300581.84].pkl")
    navigation_down_10 = ("navigation_easy_central_sudden_death_down_larger", "best_model_1930240_[-243753.58].pkl")
    navigation_up_10 = ("navigation_easy_central_sudden_death_up_larger", "best_model_453120_[-230827.05].pkl")
    model = navigation_up_10
  elif scenario == 'merging':
    safe0 = ("safe0", "eval559best_model_559_[710.741].pkl")
    eff100 = ("eff100", "eval119best_model_119_[58557.055].pkl")
    eff = ("eff", "eval489best_model_489_[-29.615425].pkl")
    curr02 = ("0.2_curr", "eval49best_model_49_[235.07114].pkl")
    curr2 = ("2.0_curr", "eval19best_model_19_[-56.795643].pkl")
    rand_start02 = ("0.2rand_start", "eval99best_model_99_[-40.224064].pkl")
    counterbalanced = ("counterbalanced", "eval9best_model_9_[-28.716702].pkl")
    weight_n1 = ("weight_-1", "best_model_151040_[710.741].pkl")
    weight_n05 = ("weight_-0.5", "best_model_1428480_[309.0573].pkl")
    weight_0 = ("weight_0", "best_model_87040_[-37.172234].pkl")
    weight_p05 = ("weight_0.5", "best_model_16640_[-33.3783].pkl")
    weight_p1d25 = ("weight_1.25", "best_model_39680_[-27.72623].pkl")
    weight_p1d5 = ("weight_1.5", "best_model_1583360_[799.3216].pkl")
    weight_p1d75 = ("weight_1.75", "best_model_112640_[942.1476].pkl")
    weight_p1 = ("weight_1", "best_model_8960_[-29.448769].pkl")
    weight_p2 = ("weight_2", "best_model_1980160_[1087.1274].pkl")
    weight_p4 = ("weight_4", "best_model_1827840_[2262.8826].pkl")
    weight_p6 = ("weight_6", "best_model_2670080_[3432.1975].pkl")
    weight_p8 = ("weight_8", "best_model_3175680_[4600.899].pkl")
    weight_p10 = ("weight_10", "best_model_3527680_[5769.4253].pkl")
    weight_p100 = ("weight_100", "best_model_14080_[58647.805].pkl")
    weight_p100 = ("weight_100_trial", "best_model_53760_[58498.3].pkl")
    weight_n1_p100 = ("weight_-1-100-residual", "best_model_107520_[-1212.9834].pkl")
    weight_n1_n05 = ("weight_-1--05-residual", "best_model_11520_[314.3503].pkl")
    weight_n1_0 = ("weight_-1-0-residual", "best_model_47360_[-43.04698].pkl")
    weight_n1_p05 = ("weight_-1-05-residual", "best_model_167680_[-33.393833].pkl")
    weight_n1_p1 = ("weight_-1-1-residual", "best_model_166400_[-29.615425].pkl")
    weight_n1_p10 = ("weight_-1-10-residual", "best_model_106240_[-137.1123].pkl")
    weight_p1_p10 = ("weight_1-10-residual", "best_model_1280_[46.919632].pkl")
    weight_p1_p100 = ("weight_1-100-residual", "best_model_1280_[808.3808].pkl")
    weight_n1_p100_fine = ("weight_-1-100-finetune", "best_model_2977280_[58541.312].pkl")
    weight_n05_p100_fine = ("weight_-05-100-finetune", "best_model_290560_[58499.133].pkl")
    weight_0_p100_fine = ("weight_0-100-finetune", "best_model_380160_[58527.918].pkl")
    weight_p05_p100_fine = ("weight_05-100-finetune", "best_model_243200_[58529.555].pkl")
    weight_p1_p100_fine = ("weight_1-100-finetune", "best_model_2447360_[58373.15].pkl")
    weight_p10_p100_fine = ("weight_10-100-finetune", "best_model_1280_[60365.332].pkl")
    weight_n1_p05_fine = ("weight_-1-0.5-finetune", "best_model_454400_[197.53082].pkl")
    weight_n1_p1d25_fine = ("weight_-1-1.25-finetune", "best_model_757760_[626.55853].pkl")
    model = weight_n1
  model_dir = os.path.join("reward_curriculum_expts", model[0], model[1])
  model = load_model(model_dir)
  eval_env = load_env()
  sum_reward = 0
  num_episode = 200
  for _ in range(num_episode):
      sum_reward += evaluate_debug(model, eval_env)
  print("mean ret: ", sum_reward/num_episode)
