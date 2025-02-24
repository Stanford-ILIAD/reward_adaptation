import os
import shutil
import gym
import time
import numpy as np
from stable_baselines import PPO2, DQN, HER, DDPG
from stable_baselines.her.utils import HERGoalEnvWrapper
import wandb
import driving_envs
import fetch.fetch_envs
import utils
from tensorflow import flags
import stable_baselines
from stable_baselines.common.vec_env import DummyVecEnv
import tensorflow as tf
from model import MlpGeneralPolicy1

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from eval_model import evaluate
import csv
import ipdb

FLAGS = flags.FLAGS


# NAVIGATION1: BARRIER SIZES
flags.DEFINE_integer("timesteps", 256000, "# timesteps to train")
flags.DEFINE_string("experiment_dir", "output/updated_gridworld_continuous2", "Name of experiment")
flags.DEFINE_string("experiment_name", "B1R_B1L2", "Name of experiment")
flags.DEFINE_boolean("is_save", True, "Saves and logs experiment data if True")
flags.DEFINE_integer("eval_save_period", 1, "how often we save state for eval")  # fine
flags.DEFINE_integer("num_envs", 1, "number of envs")
flags.DEFINE_integer("seed", 101, "random seed")
flags.DEFINE_integer("bs", 5, "barrier size")
flags.DEFINE_string("expt_type", "ours", "experiment type")

from output.updated_gridworld_continuous.policies import *

class RewardCurriculum(object):
    """
    Code related to training reward curriculum or single domain
    """

    def __init__(self, model_type, model_dir, num_envs, experiment_dir, experiment_name, timesteps, is_save,
                 eval_save_period, seed, bs, expt_type):
        self.model_type = model_type
        self.num_envs = num_envs
        self.experiment_dir1 = experiment_dir
        self.experiment_dir = os.path.join(experiment_dir, experiment_name)
        self.experiment_name = experiment_name
        self.timesteps = timesteps
        self.is_save = is_save
        self.eval_save_period = eval_save_period
        self.rets_path = None
        self.seed = seed
        self.bs = bs
        self.expt_type = expt_type
        self.curriculum = [
        ]

    def create_eval_dir(self):
        if self.is_save:
            print(self.experiment_dir)
            if os.path.exists(self.experiment_dir):
                shutil.rmtree(self.experiment_dir)
            os.makedirs(self.experiment_dir)
            self.rets_path = os.path.join(self.experiment_dir, "trajs.csv")
            #wandb.save(self.experiment_dir)

    def train_curriculum(self, env_name="Merging-v0"):
        """
        Trains reward curriculum
        """
        self.curriculum = [
            env_name
        ]
        bs2model_ours = {1:B1R_B0L, 3:B3R_B0L, 5:B5R_B0L2, 7:B7R_B0L_B4L1}
        bs2model = {1:B1R, 3:B3R, 5:B5R, 7:B7R}
        for l, lesson in enumerate(self.curriculum):
            for seed in [201, 202, 203, 204, 205]:
                if self.expt_type == "ours":
                    model_info = bs2model_ours[int(self.bs)]
                elif self.expt_type == "finetune":
                    model_info = bs2model[int(self.bs)]
                model_dir = os.path.join(model_info[0], model_info[1], model_info[2])
                if self.model_type == "PPO":
                    self.model = PPO2.load(model_dir)  # loads pre-trained model
                elif self.model_type == "HER":
                    self.model = HER.load(model_dir)   # loads pre-trained model
                print(f"\ntraining on {lesson}, bs {self.bs}, seed{seed}")
                self.seed = seed
                self.experiment_name = f"{self.bs}_{self.expt_type}_{seed}"
                print("EXPT NAME: ", self.experiment_dir1, self.experiment_name)
                self.experiment_dir = os.path.join(self.experiment_dir1, self.experiment_name)
                self.create_eval_dir()
                env = gym.make(lesson)
                eval_env = gym.make(lesson)

                env._set_barrier_size(self.bs)
                env._set_homotopy_class('left')
                eval_env._set_barrier_size(self.bs)
                eval_env._set_homotopy_class('left')

                if self.model_type == "HER":
                    env = HERGoalEnvWrapper(env)
                    eval_env = HERGoalEnvWrapper(eval_env)
                    print("bs: ", env.env.barrier_size)
                    print("hc: ", env.env.homotopy_class)
                else:
                    env = DummyVecEnv([lambda: env])
                self.model.set_env(env)
                self.model.set_random_seed(self.seed)
                ### ENTROPY###
                #self.model.ent_coef = 0.05
                self.model = train(self.model, eval_env, self.timesteps, self.experiment_dir,
                                   self.is_save, self.eval_save_period, self.rets_path, l)

    def train_single(self, env_name="Merging-v0"):
        """
        Directly trains on env_name
        """
        for seed in [201, 202, 203, 204, 205]:
            print(f"\ntraining with bsize {self.bs}, seed{seed}")
            self.seed = seed
            self.experiment_name = f"B{self.bs}R{seed}"
            print("EXPT NAME: ", self.experiment_dir1, self.experiment_name)
            self.experiment_dir = os.path.join(self.experiment_dir1, self.experiment_name)
            self.create_eval_dir()
            self.model = None
            env = gym.make(env_name)
            eval_env = gym.make(env_name)
            env._set_barrier_size(self.bs)
            env._set_homotopy_class('right')
            eval_env._set_barrier_size(self.bs)
            eval_env._set_homotopy_class('right')
            if self.model_type == "PPO":
                if self.is_save:
                    ### DEEPER NETWORK
                    #policy_kwargs = dict(net_arch=[dict(pi=[64, 64, 64, 64],
                    #                                    vf=[64, 64, 64, 64])]
                    #                                    )
                    #self.PPO = PPO2('MlpPolicy', env, verbose=1, seed=self.seed, learning_rate=1e-3,
                    #                policy_kwargs=policy_kwargs)
                    ### DROPOUT
                    #self.PPO = PPO2(MlpGeneralPolicy1, env, verbose=1, seed=self.seed, learning_rate=1e-3)
                    ### REGULAR
                    self.PPO = PPO2('MlpPolicy', env, verbose=1, seed=self.seed, learning_rate=1e-3)
                else:
                    self.PPO = PPO2('MlpPolicy', env, verbose=1, seed=self.seed, learning_rate=1e-3)

                self.model = train(self.PPO, eval_env, self.timesteps, self.experiment_dir,
                                   self.is_save, self.eval_save_period, self.rets_path, 0)
            elif self.model_type == "DQN":
                if self.is_save:
                    self.DQN = DQN('MlpPolicy', env, verbose=1, seed=self.seed, prioritized_replay=True,
                                   learning_rate=1e-3, tensorboard_log="./Gridworldv1_tensorboard/" + self.experiment_name,
                                   full_tensorboard_log=True)
                else:
                    self.DQN = DQN('MlpPolicy', env, verbose=1, seed=self.seed, prioritized_replay=True,
                                   learning_rate=1e-3)
                self.model = train(self.DQN, eval_env, self.timesteps, self.experiment_dir,
                                   self.is_save, self.eval_save_period, self.rets_path, 0)
            elif self.model_type == "HER":
                env = HERGoalEnvWrapper(env)
                eval_env = HERGoalEnvWrapper(eval_env)
                print("bs: ", env.env.barrier_size)
                print("hc: ", env.env.homotopy_class)
                self.HER = HER('MlpPolicy', env, DDPG, n_sampled_goal=4, goal_selection_strategy="future",
                               seed=self.seed, verbose=1)
                self.model = train(self.HER, eval_env, self.timesteps, self.experiment_dir,
                                   self.is_save, self.eval_save_period, self.rets_path, 0)


def train(model, eval_env, timesteps, experiment_name, is_save, eval_save_period, rets_path, num_trains):
    """
    Trains model for specified timesteps. Returns trained model.
    :param num_trains: number of previous lessons, for continual learning setting
    """

    def callback(_locals, _globals):
        nonlocal n_callbacks, best_ret
        model = _locals['self'].model
        total_steps = model.num_timesteps + (timesteps) * num_trains

        # Saving best model
        if (total_steps) % eval_save_period == 0:
            if is_save:
                ret, std, total_rets, state_history = evaluate(model, eval_env, render=False)
                # model.save(os.path.join(experiment_name, 'model_{}_{}.pkl'.format(total_steps, ret)))
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
    if FLAGS.env == 'nav1':
        #if FLAGS.is_save: wandb.init(project="continuous_updated2", sync_tensorboard=True, name=FLAGS.experiment_name)
        from output.updated_gridworld_continuous.policies import *

        if FLAGS.expt_type == "ours":
            model_info = B1R_B0L1
        else:
            model_info = B1R
        model_dir = os.path.join(model_info[0], model_info[1], model_info[2])
        RC = RewardCurriculum("PPO", model_dir, FLAGS.num_envs, FLAGS.experiment_dir, FLAGS.experiment_name,
                              FLAGS.timesteps, FLAGS.is_save, FLAGS.eval_save_period, FLAGS.seed, FLAGS.bs,
                              FLAGS.expt_type)
        if FLAGS.expt_type == "direct":
            RC.train_single(env_name="Continuous-v0")
        else:
            RC.train_curriculum(env_name="Continuous-v0")

    elif FLAGS.env == 'nav1_sparse':
        from output.updated_gridworld_continuous.policies import *
        model_info = B1R
        model_dir = os.path.join(model_info[0], model_info[1], model_info[2])
        if FLAGS.expt_type == "direct":
            RC = RewardCurriculum("HER", None, FLAGS.num_envs, FLAGS.experiment_dir, FLAGS.experiment_name,
                                  FLAGS.timesteps, FLAGS.is_save, FLAGS.eval_save_period, FLAGS.seed, FLAGS.bs,
                                  FLAGS.expt_type)
            RC.train_single(env_name="ContinuousSparse-v0")
        else:
            RC = RewardCurriculum("HER", model_dir, FLAGS.num_envs, FLAGS.experiment_dir, FLAGS.experiment_name,
                                  FLAGS.timesteps, FLAGS.is_save, FLAGS.eval_save_period, FLAGS.seed, FLAGS.bs,
                                  FLAGS.expt_type)
            RC.train_curriculum(env_name="ContinuousSparse-v0")

