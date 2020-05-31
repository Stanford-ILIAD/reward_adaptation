import os
import shutil
import gym
import time
import numpy as np
from stable_baselines import PPO2, DQN, HER, DDPG
from stable_baselines.her.utils import HERGoalEnvWrapper
import wandb
import driving.driving_envs
import fetch.fetch_envs
import utils
from tensorflow import flags
import stable_baselines
from stable_baselines.common.vec_env import DummyVecEnv
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from eval_model import evaluate
import csv
import ipdb

FLAGS = flags.FLAGS
#flags.DEFINE_integer("timesteps", 128000, "# timesteps to train")
flags.DEFINE_integer("timesteps", 512000, "# timesteps to train")  # 3000 updates
#flags.DEFINE_integer("timesteps", 160000, "# timesteps to train")
flags.DEFINE_string("experiment_dir", "output/fetch3", "Name of experiment")
flags.DEFINE_string("experiment_name", "Bn1L_seed1", "Name of experiment")
flags.DEFINE_boolean("is_save", True, "Saves and logs experiment data if True")
flags.DEFINE_integer("eval_save_period", 10000, "how often we save state for eval")
flags.DEFINE_integer("seed", 10, "random seed")
#flags.DEFINE_integer("eval_save_period", 50, "how often we save state for eval")  # fine
flags.DEFINE_integer("num_envs", 1, "number of envs")
flags.DEFINE_string("expt_type", "finetune", "expt type")
#

class RewardCurriculum(object):
    """
    Code related to training reward curriculum or single domain
    """

    def __init__(self, model_type, model_dir, num_envs, experiment_dir, experiment_name, timesteps, is_save, eval_save_period, seed):
        self.model_type = model_type
        if self.model_type == "PPO":
            self.model = PPO2.load(model_dir) if model_dir else None  # loads pre-trained model
        elif self.model_type == "DQN":
            self.model = DQN.load(model_dir) if model_dir else None  # loads pre-trained model
        elif self.model_type == "HER":
            self.model = HER.load(model_dir) if model_dir else None  # loads pre-trained model
        self.num_envs = num_envs
        self.experiment_dir = os.path.join(experiment_dir, experiment_name)
        self.experiment_name = experiment_name
        self.timesteps = timesteps
        self.is_save = is_save
        self.eval_save_period = eval_save_period
        self.rets_path = None
        self.create_eval_dir()
        #self.seed = 42
        self.seed = seed
        print("SEED: ", self.seed)
        self.curriculum = [
        ]

    def create_eval_dir(self):
        if self.is_save:
            print(self.experiment_dir)
            if os.path.exists(self.experiment_dir):
                shutil.rmtree(self.experiment_dir)
            os.makedirs(self.experiment_dir)
            self.rets_path = os.path.join(self.experiment_dir, "trajs.csv")
            wandb.save(self.experiment_dir)

    def train_curriculum(self, env_name="Merging-v0"):
        self.curriculum = [
            env_name
        ]
        """
        Trains reward curriculum
        """
        for l, lesson in enumerate(self.curriculum):
            print("\ntraining on ", lesson)
            env = gym.make(lesson)
            #env = DummyVecEnv([lambda: env])
            eval_env = gym.make(lesson)
            if self.model_type == "HER":
                env = HERGoalEnvWrapper(env)
                eval_env = HERGoalEnvWrapper(eval_env)
            self.model.set_env(env)
            #self.model.tensorboard_log = "./Gridworldv1_tensorboard/" + self.experiment_name
            #self.model.full_tensorboard_log = True
            self.model.seed = self.seed
            self.model = train(self.model, eval_env, self.timesteps, self.experiment_dir,
                               self.is_save, self.eval_save_period, self.rets_path, l)


    def train_single(self, env_name="Merging-v0"):
        """
        Directly trains on env_name
        """
        #self.timesteps = 220000 # to train for longer
        self.model = None
        env = gym.make(env_name)
        eval_env = gym.make(env_name)
        if self.model_type == "PPO":
            if self.is_save:
                self.PPO = PPO2('MlpPolicy', env, verbose=1, seed=self.seed, learning_rate=1e-3,
                                #tensorboard_log="./Gridworldv1_tensorboard/"+self.experiment_name,
                                #full_tensorboard_log=True
                                )
            else:
                self.PPO = PPO2('MlpPolicy', env, verbose=1, seed=self.seed, learning_rate=1e-3)
            self.model = train(self.PPO, eval_env, self.timesteps, self.experiment_dir,
                               self.is_save, self.eval_save_period, self.rets_path, 0)
        elif self.model_type == "DQN":
            if self.is_save:
                self.DQN = DQN('MlpPolicy', env, verbose=1, seed=self.seed, prioritized_replay=True,
                               learning_rate=1e-3, tensorboard_log="./Gridworldv1_tensorboard/"+self.experiment_name,
                               full_tensorboard_log=True)
            else:
                self.DQN = DQN('MlpPolicy', env, verbose=1, seed=self.seed, prioritized_replay=True,
                               learning_rate=1e-3)
            self.model = train(self.DQN, eval_env, self.timesteps, self.experiment_dir,
                               self.is_save, self.eval_save_period, self.rets_path, 0)
        elif self.model_type == "HER":
            env = HERGoalEnvWrapper(env)
            eval_env = HERGoalEnvWrapper(eval_env)
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
        model = _locals['self']
        total_steps = model.num_timesteps + (timesteps)*num_trains

        # Saving best model
        if (total_steps) % eval_save_period == 0:
            if is_save:
                ret, std, total_rets, state_history = evaluate(model, eval_env, render=False)
                #model.save(os.path.join(experiment_name, 'model_{}_{}.pkl'.format(total_steps, ret)))
                if ret > best_ret:
                    print("Saving new best model")
                    model.save(os.path.join(experiment_name, 'best_model_{}_{}.pkl'.format(total_steps, ret)))
                    best_ret = ret
                wandb.log({"eval_ret": ret}, step=total_steps)
                state_history = list(state_history)
                line = [total_steps] + state_history
                with open(rets_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
            else:
                print("\neval")
                print("total steps: ", total_steps)
                ret, std, total_rets, _ = evaluate(model, eval_env, render=True)
                #ret, std, total_rets, _ = evaluate(model, eval_env, render=False)
        return True
    best_ret, n_callbacks = -np.infty, 0
    model.learn(total_timesteps=timesteps, callback=callback)
    if is_save: model.save(os.path.join(experiment_name, 'final_model_{}.pkl'.format(num_trains)))
    return model


if __name__ == '__main__':
    if FLAGS.is_save: wandb.init(project="fetch_LR1", sync_tensorboard=True, name=FLAGS.experiment_name)
    #from output.updated_gridworld_continuous.policies import *
    from output.fetch2.policies import *

    if FLAGS.expt_type == "ours":
        model_info = BL_v02_BR01
        print("model info: BLv02_BR0")
    else:
        model_info = BL_v021
        print("model info: BLv021")

    model_dir = os.path.join(model_info[0], model_info[1], model_info[2])
    RC = RewardCurriculum("HER", model_dir, FLAGS.num_envs, FLAGS.experiment_dir, FLAGS.experiment_name, FLAGS.timesteps, FLAGS.is_save, FLAGS.eval_save_period, FLAGS.seed)
    if FLAGS.expt_type == "direct":
        print("train single")
        RC.train_single(env_name="Fetch-v0")
    else:
        print("train curriculum")
        RC.train_curriculum(env_name="Fetch-v0")

    #model = ('output/gridworld_continuous', 'multi_obj_policies', 'll_policy.pkl')
    #model = ('output/gridworld_continuous', 'multi_obj_policies', 'rl_policy.pkl')
    #model = ('output/gridworld_continuous', 'multi_obj_policies', 'lr_policy.pkl')
    #model = ('output/gridworld_continuous', 'multi_obj_policies', 'rr_policy.pkl')
    #RC.train_single(env_name="ContinuousNoneRL-v0")
    ##RC.train_curriculum(env_name="ContinuousNoneRL-v0")
