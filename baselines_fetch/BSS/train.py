import os
import shutil
import gym
import time
import numpy as np
import wandb
import fetch.fetch_envs
from tensorflow import flags
import stable_baselines
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from eval_model import evaluate
from stable_baselines.her.utils import HERGoalEnvWrapper
import csv

import baselines_fetch.BSS.utils as utils

from baselines_fetch.BSS.model import HER2BSS
import ipdb

FLAGS = flags.FLAGS
flags.DEFINE_integer("timesteps", 512000, "# timesteps to train")  # 3000 updates
flags.DEFINE_string("experiment_dir", "output/fetch_BSS", "Name of experiment")
flags.DEFINE_string("experiment_name", "BR_BL_BSS", "Name of experiment")
flags.DEFINE_boolean("is_save", True, "Saves and logs experiment data if True")
flags.DEFINE_integer("eval_save_period", 10000, "how often we save state for eval")
flags.DEFINE_integer("num_envs", 1, "number of envs")
flags.DEFINE_string("bs", "RL", "direction of fine-tuning")



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

    def __init__(self, model_dir, output_dir, num_envs, experiment_dir, experiment_name, timesteps, is_save, eval_save_period, bs):
        self.num_envs = num_envs
        self.experiment_dir1 = experiment_dir
        self.timesteps = timesteps
        self.is_save = is_save
        self.eval_save_period = eval_save_period
        self.rets_path = None
        self.seed = None
        self.bs = bs

    def create_eval_dir(self):
        if self.is_save:
            print(self.experiment_dir)
            if os.path.exists(self.experiment_dir):
                shutil.rmtree(self.experiment_dir)
            os.makedirs(self.experiment_dir)
            self.rets_path = os.path.join(self.experiment_dir, "trajs.csv")
            #wandb.save(self.experiment_dir)


    def train_bss(self, env_name="Merging-v0"):
        """
        Directly trains on env_name
        """

        bs2model = {'RL':BR_s, 'LR': BL_s}
        #for bs in bs2model.keys():
        bs = self.bs
        for seed in [101, 102]:
            model_info = bs2model[bs]
            model_dir = os.path.join(model_info[0], model_info[1], model_info[2])
            output_dir = os.path.join("output/fetch_BSS", 'resave', model_info[2])
            utils.resave_params_for_BSS(model_dir, output_dir)
            self.model = HER2BSS.load(output_dir)
            self.seed = seed
            self.experiment_name = f"BSS_{bs}_{self.seed}"
            print("EXPT NAME: ", self.experiment_name)
            self.experiment_dir = os.path.join(self.experiment_dir1, self.experiment_name)
            self.create_eval_dir()
            env = gym.make(env_name)
            eval_env = gym.make(env_name)
            if bs == 'RL':
                env.homotopy_class = 'left'
                eval_env.homotopy_class = 'left'
            elif bs == 'LR':
                env.homotopy_class = 'right'
                eval_env.homotopy_class = 'right'
            env = HERGoalEnvWrapper(env)
            self.model.set_env(env)
            eval_env = HERGoalEnvWrapper(eval_env)
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
                ret, std, total_rets, _ = evaluate(model, eval_env, render=True)
        return True
    best_ret, n_callbacks = -np.infty, 0
    model.learn(total_timesteps=timesteps, callback=callback)
    if is_save: model.save(os.path.join(experiment_name, 'final_model_{}.pkl'.format(num_trains)))
    return model


if __name__ == '__main__':
    #if FLAGS.is_save: wandb.init(project="fetch2", sync_tensorboard=True)
    from output.fetch2.policies import *
    model_info = BR
    model_dir, output_dir = None, None
    RC = RewardCurriculum(model_dir, output_dir, FLAGS.num_envs, FLAGS.experiment_dir, FLAGS.experiment_name,
                          FLAGS.timesteps, FLAGS.is_save, FLAGS.eval_save_period, FLAGS.bs)
    RC.train_bss(env_name="Fetch-v0")
