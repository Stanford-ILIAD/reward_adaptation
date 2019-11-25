import os
import time
import csv
import shutil
import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
import wandb
from tensorflow import flags
import driving_envs
from utils import evaluate_debug
import utils

FLAGS = flags.FLAGS
n_steps = 128
flags.DEFINE_integer("timesteps", n_steps * 100, "# timesteps to train")
# n_updates = total_timesteps/n_steps(128)
flags.DEFINE_string("name", "rew_curr", "Name of experiment")
flags.DEFINE_boolean("is_save", True, "Saves and logs experiment data if True")
flags.DEFINE_integer("eval_save_period", 10, "how often we save state for eval")


# have loop with all the environments
# switch to next environment when cumulative reward is positive

def reward_curriculum_train(model_dir, timesteps, experiment_name, is_save, eval_save_period, num_envs=1):
    """
    Trains loaded model using a curriculum.
    num_envs needs to be the same as the loaded model
    """

    # load previous model (safe)
    model = PPO2.load(model_dir)

    def callback(_locals, _globals):
        nonlocal n_steps, best_ret
        model = _locals['self']
        if (n_steps + 1) % eval_save_period == 0:
            start_eval_time = time.time()
            if is_save:
                eval_dir = os.path.join(experiment_name, "eval{}".format(n_steps))
                os.makedirs(eval_dir)
                # ret = evaluate(model, eval_dir)
                ret = evaluate_debug(model, eval_env, is_save, eval_dir)
                if ret > best_ret:
                    print("Saving new best model")
                    _locals['self'].save(eval_dir + 'best_model_{}_{}.pkl'.format(n_steps, ret))
                    best_ret = ret
            else:
                ret = evaluate_debug(model, eval_env, is_save)
            print("eval ret: ", ret)
            if is_save:
                total_steps = _locals["self"].num_timesteps + (timesteps)*l
                print("TOTAL STEPS: ", total_steps)
                wandb.log({"eval_ret": ret}, step=total_steps)
                with open(rets_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([n_steps, ret])
            end_eval_time = time.time() - start_eval_time
            print("Finished evaluation in {:.2f} seconds".format(end_eval_time))
        n_steps += 1
        return True

    curriculum = ["Merging-v2", "Merging-v3", "Merging-v4", "Merging-v5"]
    curr_params = model.get_parameters()
    for l, lesson in enumerate(curriculum):
        print("\ntraining on ", lesson)
        # change env
        env_fns = num_envs * [lambda: gym.make(lesson)]
        eval_env = VecNormalize(DummyVecEnv(env_fns), training=False, norm_reward=False)
        env = VecNormalize(SubprocVecEnv(env_fns))
        model.set_env(env)
        assert utils.check_params_equal(curr_params, model.get_parameters())

        if is_save:
            if os.path.exists(experiment_name):
                shutil.rmtree(experiment_name)
            os.makedirs(experiment_name)
            rets_path = os.path.join(experiment_name, "eval.csv")
            wandb.save(experiment_name)

        best_ret, n_steps = -np.infty, 0
        if is_save:
            eval_dir = os.path.join(experiment_name, "eval{}".format(n_steps))
            model.save(eval_dir + 'lesson_{}.pkl'.format(l+1))
        model.learn(total_timesteps=timesteps, callback=callback)
        curr_params = model.get_parameters()
        # break
    if is_save:
        eval_dir = os.path.join(experiment_name, "eval{}".format(n_steps))
        model.save(eval_dir + 'lesson_{}.pkl'.format(l+1))


if __name__ == '__main__':
    if FLAGS.is_save: wandb.init(project="reward_adaptation", sync_tensorboard=True)
    model_name = "eval559best_model_559_[710.741].pkl"
    model_dir = os.path.join("safe0", model_name)
    reward_curriculum_train(model_dir, FLAGS.timesteps, FLAGS.name, FLAGS.is_save, FLAGS.eval_save_period)
