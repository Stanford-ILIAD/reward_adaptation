import os
import shutil
import gym
import time
import numpy as np
from stable_baselines import DQN, PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common import set_global_seeds
import wandb
import minigrid.gym_minigrid
import utils
from tensorflow import flags
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from eval_model import evaluate
import csv


FLAGS = flags.FLAGS
n_steps = 128
flags.DEFINE_integer("timesteps", n_steps * 521, "# timesteps to train")
flags.DEFINE_string("name", "gridworld/v0", "Name of experiment")
flags.DEFINE_boolean("is_save", True, "Saves and logs experiment data if True")
flags.DEFINE_integer("eval_save_period", 5000, "how often we save state for eval")
flags.DEFINE_integer("num_envs", 1, "number of envs")


class RewardCurriculum(object):
    """
    Code related to training reward curriculum or single domain
    """

    def __init__(self, model_dir, num_envs, experiment_name, timesteps, is_save, eval_save_period):
        self.model = PPO2.load(model_dir) if model_dir else None  # loads pre-trained model
        self.num_envs = num_envs
        self.experiment_name = experiment_name
        self.timesteps = timesteps
        self.is_save = is_save
        self.eval_save_period = eval_save_period
        self.rets_path = None
        self.create_eval_dir()
        self.seed = 42
        self.curriculum = [
            #"Merging-v3",
            #"Merging-v4",
            #"Merging-v6",
            #"Merging-v2",
            #"Merging-v5",
            #"Merging-v7",
            #"Merging-v0",
            # "Merging-v8",
            # "Merging-v9",
            # "Merging-v10",
            "Merging-v11"
        ]

    def create_eval_dir(self):
        if self.is_save:
            print(self.experiment_name)
            if os.path.exists(self.experiment_name):
                shutil.rmtree(self.experiment_name)
            os.makedirs(self.experiment_name)
            self.rets_path = os.path.join(self.experiment_name, "eval.csv")
            wandb.save(self.experiment_name)

    def train_curriculum(self):
        """
        Trains reward curriculum
        """
        curr_params = self.model.get_parameters()
        self.timesteps = 100000
        for l, lesson in enumerate(self.curriculum):
            print("\ntraining on ", lesson)

            # change env
            env_fns = self.num_envs * [lambda: gym.make(lesson)]
            eval_env = VecNormalize(DummyVecEnv(env_fns), training=False, norm_reward=False)
            env = VecNormalize(SubprocVecEnv(env_fns))
            self.model.set_env(env)

            assert utils.check_params_equal(curr_params, self.model.get_parameters())
            self.model = train(self.model, eval_env, self.timesteps, self.experiment_name,
                               self.is_save, self.eval_save_period, self.rets_path, l)

            curr_params = self.model.get_parameters()

    def train_single(self, env_name="Merging-v0"):
        """
        Directly trains on env_name
        """
        self.timesteps = 300000 # to train for longer
        set_global_seeds(100)
        env_fns = self.num_envs * [lambda: gym.make(env_name)]
        #env = VecNormalize(SubprocVecEnv(env_fns), norm_reward=False)
        #env = VecNormalize(SubprocVecEnv(env_fns))
        env = gym.make(env_name)
        self.model = DQN('MlpPolicy', env, verbose=1, seed=self.seed, prioritized_replay=True, learning_rate=1e-3, tensorboard_log="./Gridworldv0_tensorboard/")
        #eval_env = VecNormalize(DummyVecEnv(env_fns), training=False, norm_reward=False)
        #eval_env = VecNormalize(DummyVecEnv(env_fns), training=False)
        eval_env = gym.make(env_name)
        self.model = train(self.model, eval_env, self.timesteps, self.experiment_name,
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

        # Saving checkpoint model
        #if is_save and total_steps % 1000 == 0:
        #    print("Saving checkpoint model")
        #    ret, std, total_rets = evaluate_debug(model, eval_env)
        #    model.save(os.path.join(experiment_name, "ckpt_model_{}_{}.pkl".format(total_steps, ret)))

        # Saving best model
        if (total_steps) % eval_save_period == 0:
            start_eval_time = time.time()
            if is_save:
                ret, std, total_rets = evaluate(model, eval_env)
                if ret > best_ret:
                    print("Saving new best model")
                    model.save(os.path.join(experiment_name, 'best_model_{}_{}.pkl'.format(total_steps, ret)))
                    best_ret = ret
                wandb.log({"eval_ret": ret}, step=total_steps)
                for param_name in model.get_parameters():
                    if 'deepq/model' in param_name:
                        wandb.log({param_name: wandb.Histogram(model.get_parameters()[param_name])})
                with open(rets_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([total_steps, total_rets])
            else:
                ret, std, total_rets = evaluate(model, eval_env)
            #print("eval ret: ", ret)
        #print("training steps: ", model.num_timesteps)
        return True
    best_ret, n_callbacks = -np.infty, 0
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(os.path.join(experiment_name, 'final_model_{}.pkl'.format(num_trains)))
    return model


if __name__ == '__main__':
    if FLAGS.is_save: wandb.init(project="gridworld-v0", sync_tensorboard=True)
    model_dir = os.path.join(utils.weight_n1[0], utils.weight_n1[1], utils.weight_n1[2])
    RC = RewardCurriculum(model_dir, FLAGS.num_envs, FLAGS.name, FLAGS.timesteps, FLAGS.is_save, FLAGS.eval_save_period)
    RC.train_single(env_name="Gridworld-v0")
