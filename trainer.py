import os
import time
import csv
import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
import wandb
import driving_envs
from utils import evaluate_debug


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
                ret = evaluate_debug(model, eval_env)
                if ret > best_ret:
                    print("Saving new best model")
                    model.save(os.path.join(experiment_name, 'best_model_{}_{}.pkl'.format(total_steps, ret)))
                    best_ret = ret
                wandb.log({"eval_ret": ret,
                           },
                          step=total_steps)
                #with open(rets_path, "a", newline="") as f:
                #    writer = csv.writer(f)
                #    writer.writerow([total_steps, total_rets])
            else:
                ret = evaluate_debug(model, eval_env)
            print("eval ret: ", ret)
            print("num_timesteps: ", model.num_timesteps)
            print("n_callbacks", n_callbacks)
            print("TOTAL STEPS: ", total_steps)
            end_eval_time = time.time() - start_eval_time
            print("Finished evaluation in {:.2f} seconds".format(end_eval_time))
        return True
    best_ret, n_callbacks = -np.infty, 0
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(os.path.join(experiment_name, 'final_model_{}.pkl'.format(num_trains)))
    return model

# every 128 timesteps, the callback function gets called (n_updates increases)
# every 1280 timestpes, we log the graph
