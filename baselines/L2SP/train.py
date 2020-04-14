import os
import shutil
import gym
import time
import numpy as np
import wandb
#import minigrid.gym_minigrid
import driving_envs
from tensorflow import flags
import stable_baselines
from stable_baselines.common.vec_env import DummyVecEnv
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from eval_model import evaluate
import csv

import utils

from model import PPO2L2SP

FLAGS = flags.FLAGS
flags.DEFINE_integer("timesteps", 220000, "# timesteps to train")
flags.DEFINE_string("experiment_dir", "output/gridworld_continuous", "Name of experiment")
flags.DEFINE_string("experiment_name", "B2R_B0L", "Name of experiment")
flags.DEFINE_boolean("is_save", True, "Saves and logs experiment data if True")
#flags.DEFINE_integer("eval_save_period", 30, "how often we save state for eval")
flags.DEFINE_integer("eval_save_period", 1, "how often we save state for eval")  # fine 
flags.DEFINE_integer("num_envs", 1, "number of envs")
flags.DEFINE_string("target_env", "", "Name of target environment")
flags.DEFINE_string("source_env", "", "Name of source environment")
        


def find_best(dir_name):
    def compare(item):
        return item[0]
    model_list = []
    for file_name in os.listdir(dir_name):
        if '.pkl' in file_name and ('final' not in file_name) and ('best_model' not in file_name):
            model_list.append([float(file_name.split('_')[2].split('.')[0]), int(file_name.split('_')[1]), file_name])
    best_model = max(model_list, key=compare)
    return os.path.join(dir_name, best_model[2]), best_model[1]

class RewardCurriculum(object):
    """
    Code related to training reward curriculum or single domain
    """

    def __init__(self, model_dir, num_envs, experiment_dir, experiment_name, timesteps, is_save, eval_save_period):
        data, params = utils.load_from_file(model_dir)
        self.model = PPO2L2SP.load(model_dir, original_params=params)
        self.num_envs = num_envs
        self.experiment_dir = os.path.join(experiment_dir, experiment_name)
        self.experiment_name = experiment_name
        self.timesteps = timesteps
        self.is_save = is_save
        self.eval_save_period = eval_save_period
        self.rets_path = None
        self.create_eval_dir()
        self.seed = 42

    def create_eval_dir(self):
        if self.is_save:
            print(self.experiment_dir)
            if os.path.exists(self.experiment_dir):
                shutil.rmtree(self.experiment_dir)
            os.makedirs(self.experiment_dir)
            self.rets_path = os.path.join(self.experiment_dir, "trajs.csv")
            wandb.save(self.experiment_dir)


    def train_l2sp(self, env_name="Merging-v0"):
        """
        Directly trains on env_name
        """
        #self.timesteps = 220000 # to train for longer
        env = gym.make(env_name)
        env = DummyVecEnv([lambda: env])
        self.model.set_env(env)
        eval_env = gym.make(env_name)
        self.model = train(self.model, eval_env, self.timesteps, self.experiment_dir,
                           self.is_save, self.eval_save_period, self.rets_path, 0)


def train(model, eval_env, timesteps, experiment_name, is_save, eval_save_period, rets_path, num_trains):
    """
    Trains model for specified timesteps. Returns trained model.
    :param num_trains: number of previous lessons, for continual learning setting
    """
    def callback(_locals, _globals):
        nonlocal n_callbacks, best_ret
        model = _locals['self']
        total_steps = model.num_timesteps + (timesteps)*num_trains

        # Saving best model
        if (total_steps) % eval_save_period == 0:
            start_eval_time = time.time()
            if is_save:
                ret, std, total_rets, state_history = evaluate(model, eval_env, render=False)
                model.save(os.path.join(experiment_name, 'model_{}_{}.pkl'.format(total_steps, ret)))
                if ret > best_ret:
                    print("Saving new best model")
                    model.save(os.path.join(experiment_name, 'best_model_{}_{}.pkl'.format(total_steps, ret)))
                    best_ret = ret
                wandb.log({"eval_ret": ret}, step=total_steps)
                #print("state history: ", state_history)
                #print("total_steps: ", total_steps)
                #print("writing: ", [total_steps, state_history])
                state_history = list(state_history)
                line = [total_steps] + state_history
                with open(rets_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
            else:
                ret, std, total_rets, _ = evaluate(model, eval_env, render=True)
            #print("eval ret: ", ret)
        #print("training steps: ", model.num_timesteps)
        return True
    best_ret, n_callbacks = -np.infty, 0
    model.learn(total_timesteps=timesteps, callback=callback)
    if is_save: model.save(os.path.join(experiment_name, 'final_model_{}.pkl'.format(num_trains)))
    return model


if __name__ == '__main__':
    if FLAGS.is_save: wandb.init(project="continuous", sync_tensorboard=True)
    #from output.gridworld_continuous.policies import *
    if 'LL' in FLAGS.source_env:
        model = ('output/gridworld_continuous', 'multi_obj_policies1', 'll_policy.pkl')
    elif 'RL' in FLAGS.source_env:
        model = ('output/gridworld_continuous', 'multi_obj_policies1', 'rl_policy.pkl')
    elif 'LR' in FLAGS.source_env:
        model = ('output/gridworld_continuous', 'multi_obj_policies1', 'lr_policy.pkl')
    elif 'RR' in FLAGS.source_env:
        model = ('output/gridworld_continuous', 'multi_obj_policies1', 'rr_policy.pkl')
    #model = ('output/gridworld_continuous', 'multi_obj_policies', 'rl_policy.pkl')
    #model = ('output/gridworld_continuous', 'multi_obj_policies', 'lr_policy.pkl')
    #model = ('output/gridworld_continuous', 'multi_obj_policies', 'rr_policy.pkl')
    model_dir = os.path.join(model[0], model[1], model[2])
    RC = RewardCurriculum(model_dir, FLAGS.num_envs, FLAGS.experiment_dir, FLAGS.experiment_name, FLAGS.timesteps, FLAGS.is_save, FLAGS.eval_save_period)
    RC.train_l2sp(env_name=FLAGS.target_env)
