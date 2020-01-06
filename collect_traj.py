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
from utils import *
from dtw import *

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

def evaluate_save_traj(model, eval_env, eval_path=None):
    """
    Saves trajectory of one episode
    """
    if os.path.exists(eval_path):
        os.remove(eval_path)

    rets = 0.0
    obs = eval_env.reset()
    state, ever_done = None, False
    while not ever_done:
        action, state = model.predict(obs, state=state, deterministic=True)
        # print("\naction: ", action)
        next_obs, rewards, done, _info = eval_env.step(action)
        rpos = eval_env.venv.envs[0].world.dynamic_agents[1].center
        print("rpos: ", rpos)

        with open(eval_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([rpos.x, rpos.y])

        #eval_env.render()
        if not ever_done:
            rets += rewards
        ever_done = np.logical_or(ever_done, done)
        obs = next_obs
        #time.sleep(.1)
    eval_env.close()
    return rets


def collect_trajs(models):
    for model_info in list(models.values()):
        model_dir = os.path.join(model_info[0], model_info[1], model_info[2])
        eval_path = os.path.join("trajs", model_info[1] + "_traj.csv")
        model = load_model(model_dir)
        eval_env = load_env("Merging-v0")
        evaluate_save_traj(model, eval_env, eval_path)


def read_trajs(models):
    trajs = {}
    for (name, model_info) in models.items():
        model_dir = os.path.join(model_info[0], model_info[1], model_info[2])
        eval_path = os.path.join("trajs", model_info[1] + "_traj.csv")
        trajs[name] = []
        with open(eval_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                trajs[name].append([float(row[0]), float(row[1])])
            trajs[name] = trajs[name][:-1]  # take out last position
    return trajs


def main():
    models = {"weight_-1": weight_n1, "weight_-0.5": weight_n05,
              "weight_0.5": weight_p05, "weight_1": weight_p1,
              "weight_0": weight_0, "weight_2": weight_p2,
              "weight_10": weight_p10, "weight_100": weight_p100}
    # collect trajectories
    #collect_trajs(models)
    trajs = read_trajs(models)
    for name in list(models.keys()):
        print("\n"+name)
        for name2 in list(models.keys()):
            alignment = dtw(trajs[name], trajs[name2], keep_internals=True)
            print(name2, alignment.normalizedDistance)

    ### Display the warping curve, i.e. the alignment curve
    #alignment.plot(type="threeway")

    ### Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
    #dtw(query, template, keep_internals=True,
    #    step_pattern=rabinerJuangStepPattern(6, "c")) \
    #    .plot(type="twoway", offset=-2)



if __name__ == "__main__":

    #eff = ("eff", "best_model_16640_386.66973876953125.pkl")
    #safe = ("safe", "best_model_298240_540.3814697265625.pkl")
    #safe = ("safe", "best_model_247040_257.32415771484375.pkl")
    #eff_lite = ("eff_lite", "best_model_92160_-32.7154655456543.pkl")
    #safe_lite = ("safe_lite", "best_model_1280_-56.451454162597656.pkl")
    #safe2eff = ("0.2safe_eff", "final_model_9.pkl")
    #eff2safe = ("0.2eff_safe", "final_model_9.pkl")

    #weight_n1 = ("weight_-1", "best_model_151040_[710.741].pkl")
    #weight_n05 = ("weight_-0.5", "best_model_1428480_[309.0573].pkl")
    #weight_0 = ("weight_0", "best_model_87040_[-37.172234].pkl")
    #weight_p05 = ("weight_0.5", "best_model_16640_[-33.3783].pkl")
    #weight_p1 = ("weight_1", "best_model_8960_[-29.448769].pkl")
    #weight_p2 = ("weight_2", "best_model_1980160_[1087.1274].pkl")
    #weight_p4 = ("weight_4", "best_model_1827840_[2262.8826].pkl")
    #weight_p6 = ("weight_6", "best_model_2670080_[3432.1975].pkl")
    #weight_p8 = ("weight_8", "best_model_3175680_[4600.899].pkl")
    #weight_p10 = ("weight_10", "best_model_3527680_[5769.4253].pkl")
    #weight_p100 = ("weight_100", "best_model_14080_[58647.805].pkl")

    #no_obstacle = ("no_obstacle", "best_model_680960_-2806.66162109375.pkl")
    #obstacle = ("obstacle", "best_model_5120_-3236.948974609375.pkl")
    #obstacle1 = ("obstacle1", "best_model_864000_-1840.4173583984375.pkl")
    #obstacle2 = ("obstacle2", "best_model_33280_-2025.617431640625.pkl")

    ##weight_n05 = ("weight_-0.5", "best_model_392960_7.897286891937256.pkl")
    ##weight_0 = ("weight_0", "ckpt_model_320000_131.7333221435547.pkl")
    ##weight_p1 = ("weight_1", "ckpt_model_320000_62.78840255737305.pkl")
    ##weight_p15 = ("weight_1.5", "ckpt_model_320000_131.7333221435547.pkl")
    main()



