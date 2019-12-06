import os
import time
import csv
import numpy as np
import pickle


def evaluate_debug(model, eval_env):
        """
        Evaluates model on 10 episodes of driving task.
        Returns mean episode reward and standard deviation.
        """
        total_rets = []
        for _ in range(10):
            rets = 0.0
            obs = eval_env.reset()
            state, done = None, False
            #task_data = []
            while not done:
                    action, state = model.predict(obs, state=state, deterministic=True)
                    #print("\naction", action)
                    next_obs, rewards, done, _info = eval_env.step(action)
                    #eval_env.render()
                    if not done:
                            #task_data.append([eval_env.venv.envs[0].world.state, action, rewards, done])
                            rets += rewards
                    #ever_done = np.logical_or(ever_done, done)
                    obs = next_obs
                    #time.sleep(.1)
            total_rets.append(rets)
        return np.mean(total_rets), np.std(total_rets), total_rets


def check_params_equal(param1, param2):
    """
    Checks whether two models parameters are equal
    """
    for key, val in param1.items():
        if np.any(param1[key] != param2[key]):
            return False
    return True


def rate_change_param(param1, param2):
    """
    Returns mean absolute change across parameters from param1 to param2
    """
    total_change = []
    for key, val in param1.items():
        total_change.append(np.mean(np.abs(param1[key] - param2[key])))
    return np.mean(total_change)




