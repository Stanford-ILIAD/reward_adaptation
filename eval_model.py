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
import utils

def load_env(env, num_envs=1):
    env_fns = num_envs * [lambda: gym.make(env)]
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
    nsteps = 0
    while not ever_done:
        nsteps +=1
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
    print("nsteps: ", nsteps)
    return rets


def eval_action_divergence(num_envs):
    E1 = ("weight_-1", "best_model_151040_[710.741].pkl", "Merging-v0")
    E2 = ("weight_-0.5", "best_model_1428480_[309.0573].pkl", "Merging-v2")
    E3 = ("weight_0", "best_model_87040_[-37.172234].pkl", "Merging-v3")
    E4 = ("weight_0.5", "best_model_16640_[-33.3783].pkl", "Merging-v4")
    E5 = ("weight_1", "best_model_8960_[-29.448769].pkl", "Merging-v5")
    E6 = ("weight_2", "best_model_1980160_[1087.1274].pkl", "Merging-v8")
    E7 = ("weight_4", "best_model_1827840_[2262.8826].pkl", "Merging-v9")
    E8 = ("weight_6", "best_model_2670080_[3432.1975].pkl", "Merging-v10")
    E9 = ("weight_8", "best_model_3175680_[4600.899].pkl", "Merging-v11")
    E10 = ("weight_10", "best_model_3527680_[5769.4253].pkl", "Merging-v6")
    E11 = ("weight_100", "best_model_14080_[58647.805].pkl", "Merging-v7")
    envs = [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11]
    #envs.reverse()
    envs = [E10, E5]
    #print(envs)

    for e in range(len(envs)-1):
        EP, EQ = envs[e], envs[e+1]
        print("\n", EP[0], EQ[0])

        model1_dir = os.path.join("policy_curriculum_expts", EP[0], EP[1])
        model1 = PPO2.load(model1_dir)
        model2_dir = os.path.join("policy_curriculum_expts", EQ[0], EQ[1])
        model2 = PPO2.load(model2_dir)
        env_fns1 = num_envs * [lambda: gym.make(EP[2])]
        eval_env1 = VecNormalize(DummyVecEnv(env_fns1), training=False, norm_reward=False)

        mean_KL_div = utils.evaluate_action(model1, model2, eval_env1)
        print("mean kl div: ", mean_KL_div)

        #mean_KL_div = utils.evaluate_action(model2, model1, eval_env2)
        #print("mean kl div: ", mean_KL_div)
        #break


def eval_weight_similarity(model1, model2):
    """
    Computes distance between each parameter between two models
    """
    model1_dir = os.path.join(model1[0], model1[1], model1[2])
    model1 = PPO2.load(model1_dir)
    param1 = model1.get_parameters()

    model2_dir = os.path.join(model2[0], model2[1], model2[2])
    model2 = PPO2.load(model2_dir)
    param2 = model2.get_parameters()

    for key, val in param1.items():
        print(key)
        norm1 = np.linalg.norm(param1[key]-param2[key], ord=1)
        norm2 = np.linalg.norm(param1[key]-param2[key], ord=2)
        normfro = np.linalg.norm(param1[key]-param2[key], ord='fro') if len(val.shape)>=2 else None
        norminf = np.linalg.norm(param1[key]-param2[key], ord=np.inf)
        print(norm1, norm2, normfro, norminf)



#def evaluate(model_dir, num_envs=1):
#    """
#    Evaluates model on one episode of driving task. Returns mean episode reward.
#    """
#
#    env_fns = num_envs * [lambda: gym.make("Merging-v0")]
#    eval_env = VecNormalize(DummyVecEnv(env_fns), training=False, norm_reward=False)
#    env = VecNormalize(SubprocVecEnv(env_fns))
#    policy = MlpPolicy
#    model = PPO2(policy, env, verbose=1)
#    PPO2.load(model_dir)
#
#    num_iters = 3
#    for _ in range(num_iters):
#        obs = eval_env.reset()
#        rets = np.zeros(num_envs)
#        state, dones = None, [False for _ in range(num_envs)]
#        ever_done = np.zeros((num_envs,), dtype=np.bool)
#        task_data = collections.defaultdict(list)  # Maps env_idx -> (state, action, reward, done) tuples
#        while not np.all(ever_done):
#            true_states = [
#                inner_env.world.state for inner_env in eval_env.venv.envs
#            ]
#            action, state = model.predict(obs, state=state, mask=dones, deterministic=True)
#            next_obs, rewards, dones, _info = eval_env.step(action)
#            eval_env.render()
#            for env_idx, data in enumerate(zip(true_states, action, rewards, dones)):
#                if not ever_done[env_idx]:
#                    task_data[env_idx].append(data)
#                    rets[env_idx] += rewards[env_idx]
#            ever_done = np.logical_or(ever_done, dones)
#            obs = next_obs
#            time.sleep(.1)
#    return np.mean(rets / num_iters)


if __name__ == "__main__":
    n1 = ("sample_complexity", "-1", "best_model_192000_-32.513710021972656.pkl")
    model = n1
    model_dir = os.path.join(model[0], model[1], model[2])
    eval_env = load_env("Merging-v0")

    #model = n1_p10
    #model_dir = os.path.join("merging", model[0], model[1])
    #eval_env = load_env("Merging-v6")

    model = load_model(model_dir)
    sum_reward = 0
    num_episode = 200
    for _ in range(num_episode):
        sum_reward += evaluate_debug(model, eval_env)
    print("mean ret: ", sum_reward/num_episode)
