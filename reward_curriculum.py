import os
import shutil
import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
import wandb
from tensorflow import flags
import driving_envs
import utils
from trainer import train

FLAGS = flags.FLAGS
n_steps = 128
flags.DEFINE_integer("timesteps", n_steps * 100, "# timesteps to train")
# n_updates = total_timesteps/n_steps(128)
flags.DEFINE_string("name", "reward_curriculum_expts/eff_lite", "Name of experiment")
flags.DEFINE_boolean("is_save", True, "Saves and logs experiment data if True")
flags.DEFINE_integer("eval_save_period", 10, "how often we save state for eval")
flags.DEFINE_integer("num_envs", 1, "number of envs")
flags.DEFINE_string("env_name", "Merging-v0", "The name of the environment.")


class RewardCurriculum(object):
    """
    Code related to training reward curriculum or single domain
    """

    def __init__(self, model_dir, num_envs, experiment_name, timesteps, is_save, eval_save_period):
        if model_dir is not None:
          self.model = PPO2.load(model_dir)  # loads pre-trained model
        self.num_envs = num_envs
        self.experiment_name = experiment_name
        self.timesteps = timesteps
        self.is_save = is_save
        self.eval_save_period = eval_save_period
        self.eval_dir = None
        self.create_eval_dir()
        self.curriculum = [
            "Merging-v2",
            "Merging-v3",
            "Merging-v4",
            "Merging-v5",
            "Merging-v6",
            "Merging-v7",
            "Merging-v8",
            "Merging-v9",
            "Merging-v10",
            "Merging-v11"
        ]

    def create_eval_dir(self):
        if self.is_save:
            if os.path.exists(self.experiment_name):
                shutil.rmtree(self.experiment_name)
            os.makedirs(self.experiment_name)
            wandb.save(self.experiment_name)

    def train_curriculum(self):
        """
        Trains reward curriculum
        """
        curr_params = self.model.get_parameters()
        for l, lesson in enumerate(self.curriculum):
            print("\ntraining on ", lesson)

            # change env
            env_fns = self.num_envs * [lambda: gym.make(lesson)]
            eval_env = VecNormalize(DummyVecEnv(env_fns), training=False, norm_reward=False)
            env = VecNormalize(SubprocVecEnv(env_fns))
            self.model.set_env(env)

            assert utils.check_params_equal(curr_params, self.model.get_parameters())
            self.model = train(self.model, eval_env, self.timesteps, self.experiment_name,
                               self.is_save, self.eval_save_period, l)

            curr_params = self.model.get_parameters()

    def train_single(self):
        """
        Directly trains on single domain
        """
        self.timesteps = 4000000  # to train for longer
        env_fns = self.num_envs * [lambda: gym.make('Merging-v7')]
        env = VecNormalize(SubprocVecEnv(env_fns))
        policy = MlpPolicy
        self.model = PPO2(policy, env, verbose=1)
        eval_env = VecNormalize(DummyVecEnv(env_fns), training=False, norm_reward=False)
        self.model = train(self.model, eval_env, self.timesteps, self.experiment_name,
                           self.is_save, self.eval_save_period, 0)


if __name__ == '__main__':
    if FLAGS.is_save: wandb.init(project="reward_adaptation20", sync_tensorboard=True, entity='caozj')
    model_name = "best_model_151040_[710.741].pkl"
    model_dir = os.path.join("reward_curriculum_expts", "weight_-1", model_name)
    RC = RewardCurriculum(model_dir, FLAGS.num_envs, FLAGS.name, FLAGS.timesteps, FLAGS.is_save, FLAGS.eval_save_period)
    #RC.train_curriculum()
    RC.train_single()
