import os
import time
import csv
import numpy as np
import pickle


def evaluate_debug(model, eval_env, need_traj=False):
    """
        Evaluates model on one episode of driving task. Returns mean episode reward.
        """
    rets = 0.0
    obs = eval_env.reset()
    state, ever_done = None, False
    task_data = []
    if need_traj:
        traj = []
    while not ever_done:
        traj.append([obs[0][0], obs[0][1]])
        action, state = model.predict(obs, state=state, deterministic=True)
        next_obs, rewards, done, _info = eval_env.step(action)
        #eval_env.render()
        if not ever_done:
            task_data.append([eval_env.venv.envs[0].world.state, action, rewards, done])
            rets += rewards
        ever_done = np.logical_or(ever_done, done)
        obs = next_obs
        # time.sleep(.1)
    if need_traj:
        return rets, traj
    return rets


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
