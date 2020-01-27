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
from utils import evaluate_debug


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
                ret, std, total_rets = evaluate_debug(model, eval_env)
                if ret > best_ret:
                    print("Saving new best model")
                    model.save(os.path.join(experiment_name, 'best_model_{}_{}.pkl'.format(total_steps, ret)))
                    best_ret = ret
                wandb.log({"eval_ret": ret,
                           },
                          step=total_steps)
                with open(rets_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([total_steps, total_rets])
            else:
                ret, std, total_rets = evaluate_debug(model, eval_env)
            #print("eval ret: ", ret)
        #print("training steps: ", model.num_timesteps)
        return True
    best_ret, n_callbacks = -np.infty, 0
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(os.path.join(experiment_name, 'final_model_{}.pkl'.format(num_trains)))
    return model

# (for DQN) every timestep, the callback function gets called (n_updates increases)
# every 1280 timestpes, we log the graph
