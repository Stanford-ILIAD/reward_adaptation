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


def train(model, eval_env, timesteps, experiment_name, is_save, eval_save_period, num_trains):
    """
    Trains model for specified timesteps. Returns trained model.
    :param num_trains: number of previous lessons, for continual learning setting
    """
    def callback(_locals, _globals):
        nonlocal n_callbacks, best_ret
        n_callbacks += 1
        model = _locals['self']
        total_steps = model.num_timesteps + (timesteps)*num_trains
        if (n_callbacks) % eval_save_period == 0:
            start_eval_time = time.time()
            if is_save:
                #eval_dir = os.path.join(experiment_name, "eval{}".format(total_steps))
                #os.makedirs(eval_dir)
                # ret = evaluate(model, eval_dir)
                ret = evaluate_debug(model, eval_env)
                if ret > best_ret:
                    print("Saving new best model")
                    model.save(os.path.join(experiment_name, 'best_model_{}_{}.pkl'.format(total_steps, ret)))
                    best_ret = ret
            else:
                ret = evaluate_debug(model, eval_env, is_save)
            print("eval ret: ", ret)
            print("num_timesteps: ", model.num_timesteps)
            print("n_callbacks", n_callbacks)
            print("TOTAL STEPS: ", total_steps)
            if is_save:
                wandb.log({"eval_ret": ret,
                           },
                          step=total_steps)
                #with open(rets_path, "a", newline="") as f:
                #    writer = csv.writer(f)
                #    writer.writerow([n_steps, ret])
            end_eval_time = time.time() - start_eval_time
            print("Finished evaluation in {:.2f} seconds".format(end_eval_time))
        return True

    #if is_save:
    #    if os.path.exists(experiment_name):
    #        shutil.rmtree(experiment_name)
    #    os.makedirs(experiment_name)
    #    #rets_path = os.path.join(experiment_name, "eval.csv")
    #    wandb.save(experiment_name)

    best_ret, n_callbacks = -np.infty, 0
    #if is_save:
    #    eval_dir = os.path.join(experiment_name, "eval{}".format(n_steps))
    #    model.save(eval_dir + 'lesson_{}.pkl'.format(l+1))
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(os.path.join(experiment_name, 'final_model_{}.pkl'.format(num_trains)))

    #if is_save:
    #    eval_dir = os.path.join(experiment_name, "eval{}".format(n_steps))
    #    model.save(eval_dir + 'lesson_{}.pkl'.format(l+1))

    return model

