import os
import shutil
import gym
import time
import numpy as np
from stable_baselines import PPO2, DQN
import lunarlander.custom_lunarlander_envs
from tensorflow import flags
import stable_baselines
from stable_baselines.common.vec_env import DummyVecEnv
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import csv

FLAGS = flags.FLAGS
flags.DEFINE_integer("timesteps", 300000, "# timesteps to train")
flags.DEFINE_string("experiment_dir", "output", "Name of experiment")
flags.DEFINE_string("experiment_name", "L", "Name of experiment")
flags.DEFINE_boolean("is_save", True, "Saves and logs experiment data if True")
flags.DEFINE_integer("eval_save_period", 1, "how often we save state for eval")  # fine 
flags.DEFINE_integer("num_envs", 1, "number of envs")
flags.DEFINE_string("target_env", "", "Name of target environment")
flags.DEFINE_string("source_env", "", "Name of source environment")
flags.DEFINE_integer("seed", 2345, "random seed")
        
def evaluate(model, eval_env, render=False):
    """
    Evaluates model on 10 episodes of driving task.
    Returns mean episode reward and standard deviation.
    """
    total_rets = []
    state_history = []
    for e in range(2):
        nsteps = 0
        rets = 0.0
        obs = eval_env.reset()
        state_history.append(obs[:2])
        state, ever_done = None, False
        while not ever_done:
            nsteps += 1
            action, state = model.predict(obs, state=state, deterministic=True)
            next_obs, ret, done, _info = eval_env.step(action, verbose=render)
            # print("ret: ", ret)
            if render: eval_env.render()
            if not ever_done:
                rets += ret
            # print("rets: ", rets)
            obs = next_obs
            #state_history.append(obs[:2])
            if render: time.sleep(.1)
            ever_done = done
            if nsteps > 3000:
                print(nsteps)
            if nsteps > 5000:
                break
        total_rets.append(rets)
        print("total mean ep return: ", np.mean(total_rets), total_rets)
        print("nsteps: ", nsteps)
    return np.mean(total_rets), np.std(total_rets), total_rets, np.array(state_history)

def find_best(dir_name):
    def compare(item):
        return item[0]
    model_list = []
    for file_name in os.listdir(dir_name):
        if '.pkl' in file_name and ('final' not in file_name) and ('best_model' not in file_name):
            model_list.append([float(file_name.split('_')[2].split('.')[0]), int(file_name.split('_')[1]), file_name])
    sorted_model_list = sorted(model_list, key=compare)
    for item in sorted_model_list:
        if item[1] >= 300 or item[1] >= sorted_model_list[-1][1]-100.:
            return_item = item
            break
    return os.path.join(dir_name, return_item[2]), return_item[1]

class RewardCurriculum(object):
    """
    Code related to training reward curriculum or single domain
    """

    def __init__(self, model_type, model_dir, num_envs, experiment_dir, experiment_name, timesteps, is_save, eval_save_period, seed=2345):
        self.model_type = "PPO"
        if self.model_type == "PPO":
            self.model = PPO2.load(model_dir) if model_dir else None  # loads pre-trained model
        elif self.model_type == "DQN":
            self.model = DQN.load(model_dir) if model_dir else None  # loads pre-trained model
        self.num_envs = num_envs
        self.experiment_dir = os.path.join(experiment_dir, experiment_name)
        self.experiment_name = experiment_name
        self.timesteps = timesteps
        self.is_save = is_save
        self.eval_save_period = eval_save_period
        self.rets_path = None
        self.create_eval_dir()
        self.seed = seed
        self.curriculum = [
        ]

    def create_eval_dir(self):
        if self.is_save:
            print(self.experiment_dir)
            if os.path.exists(self.experiment_dir):
                shutil.rmtree(self.experiment_dir)
            os.makedirs(self.experiment_dir)
            self.rets_path = os.path.join(self.experiment_dir, "trajs.csv")
    def create_eval_dir1(self, experiment_dir):
        if self.is_save:
            print(experiment_dir)
            if os.path.exists(experiment_dir):
                shutil.rmtree(experiment_dir)
            os.makedirs(experiment_dir)
            self.rets_path = os.path.join(experiment_dir, "trajs.csv")

    def train_curriculum(self, env_name="Merging-v0", size_list=[], r_list=[]):
        self.curriculum = [
            env_name
        ]
        """
        Trains reward curriculum
        """
        #curr_params = self.model.get_parameters()
        self.timesteps = 300000 if len(size_list) == 0 and len(r_list) == 0 else 100000 # to train for longer
        for l, lesson in enumerate(self.curriculum):
            if len(r_list) > 0:
                jj = 0
                iter_nums = 0
                for r_weight in r_list:
                    print("\ntraining on ", lesson)

                    env = gym.make(lesson)
                    env.set_factor(r_weight)
                    env.set_obs('rec', [-0.05, 0.05, 1/6., 1/3.])
                    env = DummyVecEnv([lambda: env])
                    self.model.set_env(env)
                    eval_env = gym.make(lesson)
                    eval_env.set_factor(r_weight)
                    eval_env.set_obs('rec', [-0.05, 0.05, 1/6., 1/3.])
                    self.create_eval_dir1(self.experiment_dir+'_step_{:02d}'.format(jj))
                    ret, _, _, _ = evaluate(self.model, eval_env, render=False)
                    self.model.save(os.path.join(self.experiment_dir+'_step_{:02d}'.format(jj), 'model_{}_{}.pkl'.format(0, ret)))
                    self.model = train(self.model, eval_env, self.timesteps, self.experiment_dir+'_step_{:02d}'.format(jj),
                               self.is_save, self.eval_save_period, self.rets_path, l)
                    best_model, iter_num = find_best(self.experiment_dir+'_step_{:02d}'.format(jj))
                    if self.model_type == "PPO":
                        self.model = PPO2.load(best_model)
                    elif self.model_type == "DQN":
                        self.model = DQN.load(best_model)
                    iter_nums += iter_num
                    jj += 1
            elif len(size_list) > 0:
                jj = 0
                iter_nums = 0
                for obs_size in size_list:
                    print("\ntraining on ", lesson)

                    env = gym.make(lesson)
                    env.set_obs('rec', [-obs_size, obs_size, 1/6., 1/3.])
                    env = DummyVecEnv([lambda: env])
                    self.model.set_env(env)
                    eval_env = gym.make(lesson)
                    eval_env.set_obs('rec', [-obs_size, obs_size, 1/6., 1/3.])
                    self.create_eval_dir1(self.experiment_dir+'_step_{:02d}'.format(jj))
                    ret, _, _, _ = evaluate(self.model, eval_env, render=False)
                    self.model.save(os.path.join(self.experiment_dir+'_step_{:02d}'.format(jj), 'model_{}_{}.pkl'.format(0, ret)))
                    self.model = train(self.model, eval_env, self.timesteps, self.experiment_dir+'_step_{:02d}'.format(jj),
                               self.is_save, self.eval_save_period, self.rets_path, l)
                    best_model, iter_num = find_best(self.experiment_dir+'_step_{:02d}'.format(jj))
                    if self.model_type == "PPO":
                        self.model = PPO2.load(best_model)
                    elif self.model_type == "DQN":
                        self.model = DQN.load(best_model)
                    iter_nums += iter_num
                    jj += 1
            else:
                print("\ntraining on ", lesson)
                env = gym.make(lesson)
                #env.set_obs('rec', [-0.05, 0.05, 1/6., 1/3.])
                env = DummyVecEnv([lambda: env])
                self.model.set_env(env)
                eval_env = gym.make(lesson)
                #eval_env.set_obs('rec', [-0.05, 0.05, 1/6., 1/3.])
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

        env.set_obs('rec', [-0.05,0.05,1./6,1./3.])
        eval_env.set_obs('rec', [-0.05,0.05,1./6.,1./3.])

        if self.model_type == "PPO":
            if self.is_save:
                self.PPO = PPO2('MlpPolicy', env, verbose=1, seed=self.seed, learning_rate=1e-3,
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
            else:
                ret, std, total_rets, _ = evaluate(model, eval_env, render=True)
        return True
    best_ret, n_callbacks = -np.infty, 0
    model.learn(total_timesteps=timesteps, callback=callback)
    if is_save: model.save(os.path.join(experiment_name, 'final_model_{}.pkl'.format(num_trains)))
    return model

if __name__ == '__main__':
    if FLAGS.source_env == "":
        model_dir = None
    elif 'LObstacle' in FLAGS.target_env:
        model = ('pretrained_policies', 'l_none.pkl')
        model_dir = os.path.join(model[0], model[1])
    elif 'RObstacle' in FLAGS.target_env:
        model = ('pretrained_policies', 'r_none.pkl')
        model_dir = os.path.join(model[0], model[1])
    elif 'L' in FLAGS.target_env:
        model = ('pretrained_policies', 'r.pkl')
        model_dir = os.path.join(model[0], model[1])
    elif 'R' in FLAGS.target_env:
        model = ('pretrained_policies', 'l.pkl')
        model_dir = os.path.join(model[0], model[1])
    RC = RewardCurriculum("PPO", model_dir, FLAGS.num_envs, FLAGS.experiment_dir, FLAGS.experiment_name, FLAGS.timesteps, FLAGS.is_save, FLAGS.eval_save_period, seed=FLAGS.seed)
    if model_dir is None:        
        RC.train_single(env_name=FLAGS.target_env)
    else:
        if 'obs_size' in FLAGS.experiment_name:
            RC.train_curriculum(env_name=FLAGS.target_env, size_list=[0.01,0.02,0.03,0.05])
        elif 'reward_w' in FLAGS.experiment_name:
            RC.train_curriculum(env_name=FLAGS.target_env, r_list=[0.1,0.4,0.7,1.])
        else:
            RC.train_curriculum(env_name=FLAGS.target_env)
