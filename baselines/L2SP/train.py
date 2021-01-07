import os
import shutil
import gym
import time
import numpy as np
import wandb
#import minigrid.gym_minigrid
import driving.driving_envs
from tensorflow import flags
import stable_baselines
from stable_baselines.common.vec_env import DummyVecEnv
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from eval_model import evaluate
import csv

import utils

from baselines.L2SP.model import PPO2L2SP
import ipdb

FLAGS = flags.FLAGS
flags.DEFINE_integer("timesteps", 256000, "# timesteps to train")
flags.DEFINE_string("experiment_dir", "output/updated_gridworld_continuous_L2SP", "Name of experiment")
flags.DEFINE_string("experiment_name", "B0R_B0L_L2SP", "Name of experiment")
flags.DEFINE_boolean("is_save", True, "Saves and logs experiment data if True")
flags.DEFINE_integer("eval_save_period", 1, "how often we save state for eval")  # fine 
flags.DEFINE_integer("num_envs", 1, "number of envs")
flags.DEFINE_integer("seed", 10, "random seed")
flags.DEFINE_string("target_env", "", "Name of target environment")
flags.DEFINE_string("source_env", "", "Name of source environment")
flags.DEFINE_integer("bs", 1, "barrier_size")
        


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
    def __init__(self, model_dir, num_envs, experiment_dir, experiment_name, timesteps, is_save, eval_save_period, seed, bs):
        #data, params = utils.load_from_file(model_dir)
        #self.model = PPO2L2SP.load(model_dir, original_params=params)
        self.num_envs = num_envs
        self.experiment_dir1 = experiment_dir
        self.experiment_dir = os.path.join(experiment_dir, experiment_name)
        self.experiment_name = experiment_name
        self.timesteps = timesteps
        self.is_save = is_save
        self.eval_save_period = eval_save_period
        self.rets_path = None
        #self.create_eval_dir()
        self.seed = seed
        self.bs = bs

    def create_eval_dir(self):
        if self.is_save:
            print(self.experiment_dir)
            if os.path.exists(self.experiment_dir):
                shutil.rmtree(self.experiment_dir)
            os.makedirs(self.experiment_dir)
            self.rets_path = os.path.join(self.experiment_dir, "trajs.csv")
            #wandb.save(self.experiment_dir)


    def train_l2sp(self, env_name="Merging-v0"):
        """
        Directly trains on env_name
        """
        bs2model = {1:B1R, 3:B3R, 5:B5R, 7:B7R}
        model_info = bs2model[int(self.bs)]
        model_dir = os.path.join(model_info[0], model_info[1], model_info[2])
        data, params = utils.load_from_file(model_dir)
        self.model = PPO2L2SP.load(model_dir, original_params=params)
        for seed in [201, 202, 203, 204, 205]:
                self.seed = seed
                self.experiment_name = f"{model_info[1]}_B{self.bs}L_L2SP{seed}"
                print("EXPT NAME: ", self.experiment_name)
                self.experiment_dir = os.path.join(self.experiment_dir1, self.experiment_name)
                self.create_eval_dir()
                env = gym.make(env_name)
                env.barrier_size = self.bs
                env = DummyVecEnv([lambda: env])
                self.model.set_env(env)
                eval_env = gym.make(env_name)
                eval_env.barrier_size = self.bs
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
        n_callbacks += 1
        #total_steps = model.num_timesteps + (timesteps)*num_trains
        total_steps = n_callbacks * model.n_steps
        print("total steps: ", total_steps)

        # Saving best model
        if (total_steps) % eval_save_period == 0:
            start_eval_time = time.time()
            if is_save:
                ret, std, total_rets, state_history = evaluate(model, eval_env, render=False)
                #model.save(os.path.join(experiment_name, 'model_{}_{}.pkl'.format(total_steps, ret)))
                if ret > best_ret:
                    print("Saving new best model")
                    model.save(os.path.join(experiment_name, 'best_model_{}_{}.pkl'.format(total_steps, ret)))
                    best_ret = ret
                #wandb.log({"eval_ret": ret}, step=total_steps)
                state_history = list(state_history)
                line = [total_steps] + state_history
                with open(rets_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
            else:
                ret, std, total_rets, _ = evaluate(model, eval_env, render=False)
        return True
    best_ret, n_callbacks = -np.infty, 0
    model.learn(total_timesteps=timesteps, callback=callback)
    if is_save: model.save(os.path.join(experiment_name, 'final_model_{}.pkl'.format(num_trains)))
    return model


if __name__ == '__main__':
    #if FLAGS.is_save: wandb.init(project="continuous_updated2", sync_tensorboard=True, name=FLAGS.experiment_name)
    from output.updated_gridworld_continuous.policies import *
    model = B1R2
    model_dir = os.path.join(model[0], model[1], model[2])
    RC = RewardCurriculum(model_dir, FLAGS.num_envs, FLAGS.experiment_dir, FLAGS.experiment_name, FLAGS.timesteps,
            FLAGS.is_save, FLAGS.eval_save_period, FLAGS.seed, FLAGS.bs)
    RC.train_l2sp(env_name="Continuous-v0")
