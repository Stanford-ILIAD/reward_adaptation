import os
import shutil
import gym
import time
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
import wandb
import driving_envs
import utils
from trainer import train
from tensorflow import flags
import tensorflow as tf

FLAGS = flags.FLAGS
n_steps = 128
#flags.DEFINE_integer("timesteps", n_steps * 781, "# timesteps to train")
flags.DEFINE_integer("timesteps", n_steps * 521, "# timesteps to train")
# n_updates = total_timesteps/n_steps(128)
flags.DEFINE_string("name", "omg2/weight_-1", "Name of experiment")
flags.DEFINE_boolean("is_save", True, "Saves and logs experiment data if True")
flags.DEFINE_integer("eval_save_period", 10, "how often we save state for eval")
flags.DEFINE_integer("num_envs", 1, "number of envs")


class PolicyCurriculum(object):
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
        self.curriculum = [
            #"Merging-v3",
            #"Merging-v4",
            #"Merging-v6",
            #"Merging-v2",
            #"Merging-v5",
            #"Merging-v7",
            "Merging-v0",
            # "Merging-v8",
            # "Merging-v9",
            # "Merging-v10",
            # "Merging-v11"
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
        self.timesteps = 4000000  # to train for longer
        for l, lesson in enumerate(self.curriculum):
            print("\ntraining on ", lesson)

            # change env
            env_fns = self.num_envs * [lambda: gym.make(lesson)]
            eval_env = VecNormalize(DummyVecEnv(env_fns), training=False, norm_reward=False)
            env = VecNormalize(SubprocVecEnv(env_fns))
            print("MODEL'S AS: ", self.model.action_space)
            print(self.model.action_space == env.action_space)
            self.model.set_env(env)

            assert utils.check_params_equal(curr_params, self.model.get_parameters())
            self.model = train(self.model, eval_env, self.timesteps, self.experiment_name,
                               self.is_save, self.eval_save_period, self.rets_path, l)

            curr_params = self.model.get_parameters()

    def train_single(self, env_name="Merging-v0"):
        """
        Directly trains on env_name
        """
        self.timesteps = 4000000 # to train for longer
        env_fns = self.num_envs * [lambda: gym.make(env_name)]
        env = VecNormalize(SubprocVecEnv(env_fns))
        policy = MlpPolicy
        self.model = PPO2(policy, env, verbose=1)
        eval_env = VecNormalize(DummyVecEnv(env_fns), training=False, norm_reward=False)
        self.model = train(self.model, eval_env, self.timesteps, self.experiment_name,
                           self.is_save, self.eval_save_period, self.rets_path, 0)

    #def eval_action_divergence(self):
    #    E1 = ("weight_-1", "best_model_151040_[710.741].pkl", "Merging-v0")
    #    E2 = ("weight_-0.5", "best_model_1428480_[309.0573].pkl", "Merging-v2")
    #    E3 = ("weight_0", "best_model_87040_[-37.172234].pkl", "Merging-v3")
    #    E4 = ("weight_0.5", "best_model_16640_[-33.3783].pkl", "Merging-v4")
    #    E5 = ("weight_1", "best_model_8960_[-29.448769].pkl", "Merging-v5")
    #    E6 = ("weight_2", "best_model_1980160_[1087.1274].pkl", "Merging-v8")
    #    E7 = ("weight_4", "best_model_1827840_[2262.8826].pkl", "Merging-v9")
    #    E8 = ("weight_6", "best_model_2670080_[3432.1975].pkl", "Merging-v10")
    #    E9 = ("weight_8", "best_model_3175680_[4600.899].pkl", "Merging-v11")
    #    E10 = ("weight_10", "best_model_3527680_[5769.4253].pkl", "Merging-v6")
    #    E11 = ("weight_100", "best_model_14080_[58647.805].pkl", "Merging-v7")
    #    envs = [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11]
    #    #envs.reverse()
    #    envs = [E10, E5]
    #    #print(envs)

    #    for e in range(len(envs)-1):
    #        EP, EQ = envs[e], envs[e+1]
    #        print("\n", EP[0], EQ[0])

    #        model1_dir = os.path.join("policy_curriculum_expts", EP[0], EP[1])
    #        model1 = PPO2.load(model1_dir)
    #        model2_dir = os.path.join("policy_curriculum_expts", EQ[0], EQ[1])
    #        model2 = PPO2.load(model2_dir)
    #        env_fns1 = self.num_envs * [lambda: gym.make(EP[2])]
    #        eval_env1 = VecNormalize(DummyVecEnv(env_fns1), training=False, norm_reward=False)

    #        mean_KL_div = utils.evaluate_action(model1, model2, eval_env1)
    #        print("mean kl div: ", mean_KL_div)

    #        #mean_KL_div = utils.evaluate_action(model2, model1, eval_env2)
    #        #print("mean kl div: ", mean_KL_div)
    #        #break

    #def eval_weight_similarity(self):
    #    E1 = ("weight_-1", "best_model_151040_[710.741].pkl", "Merging-v0")
    #    E2 = ("weight_-0.5", "best_model_1428480_[309.0573].pkl", "Merging-v2")
    #    E3 = ("weight_0", "best_model_87040_[-37.172234].pkl", "Merging-v3")
    #    E4 = ("weight_0.5", "best_model_16640_[-33.3783].pkl", "Merging-v4")
    #    E5 = ("weight_1", "best_model_8960_[-29.448769].pkl", "Merging-v5")
    #    E6 = ("weight_2", "best_model_1980160_[1087.1274].pkl", "Merging-v8")
    #    E7 = ("weight_4", "best_model_1827840_[2262.8826].pkl", "Merging-v9")
    #    E8 = ("weight_6", "best_model_2670080_[3432.1975].pkl", "Merging-v10")
    #    E9 = ("weight_8", "best_model_3175680_[4600.899].pkl", "Merging-v11")
    #    E10 = ("weight_10", "best_model_3527680_[5769.4253].pkl", "Merging-v6")
    #    E11 = ("weight_100", "best_model_14080_[58647.805].pkl", "Merging-v7")

    #    model1_dir = os.path.join("policy_curriculum_expts", E2[0], E2[1])
    #    model1 = PPO2.load(model1_dir)
    #    param1 = model1.get_parameters()

    #    model2_dir = os.path.join("policy_curriculum_expts", E1[0], E1[1])
    #    model2 = PPO2.load(model2_dir)
    #    param2 = model2.get_parameters()

    #    for key, val in param1.items():
    #        print(key)
    #        norm2 = np.linalg.norm(param1[key]-param2[key], ord=2)
    #        normfro = np.linalg.norm(param1[key]-param2[key], ord='fro') if len(val.shape)>=2 else None
    #        norminf = np.linalg.norm(param1[key]-param2[key], ord=np.inf)
    #        print(norm2, normfro, norminf)


    #    model2_dir = os.path.join("policy_curriculum_expts", E11[0], E11[1])
    #    model2 = PPO2.load(model2_dir)
    #    param2 = model2.get_parameters()

    #    print()
    #    for key, val in param1.items():
    #        print(key)
    #        norm2 = np.linalg.norm(param1[key]-param2[key], ord=2)
    #        normfro = np.linalg.norm(param1[key]-param2[key], ord='fro') if len(val.shape)>=2 else None
    #        norminf = np.linalg.norm(param1[key]-param2[key], ord=np.inf)
    #        print(norm2, normfro, norminf)




if __name__ == '__main__':
    if FLAGS.is_save: wandb.init(project="restricted_heading", sync_tensorboard=True)
    weight_n05 = ("weight_-0.5", "best_model_392960_7.897286891937256.pkl")
    weight_0 = ("weight_0", "ckpt_model_320000_131.7333221435547.pkl")
    weight_p1 = ("weight_1", "ckpt_model_320000_62.78840255737305.pkl")
    weight_p15 = ("weight_1.5", "ckpt_model_320000_131.7333221435547.pkl")
    model = weight_p1
    model_dir = os.path.join("omg", model[0], model[1])
    PC = PolicyCurriculum(model_dir, FLAGS.num_envs, FLAGS.name, FLAGS.timesteps, FLAGS.is_save, FLAGS.eval_save_period)
    #PC.train_curriculum()
    PC.train_single("Merging-v0")

    #lesson = "Merging-v4"
    #PC.experiment_name = FLAGS.name + lesson
    #PC.create_eval_dir()
    #PC.train_single(lesson)

    # print("curr: ", PC.curriculum)
    # for l, lesson in enumerate(PC.curriculum):
    #    PC.experiment_name = FLAGS.name + lesson
    #    PC.create_eval_dir()
    #    print("training: ", PC.experiment_name)
    #    PC.train_single(lesson)
