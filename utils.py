import os
import time
import csv
import numpy as np
import pickle


def evaluate_debug(model, eval_env, is_save, eval_dir=None):
        """
        Evaluates model on one episode of driving task. Returns mean episode reward.
        """
        rets = 0.0
        obs = eval_env.reset()
        state, ever_done = None, False
        task_data = []
        while not ever_done:
                action, state = model.predict(obs, state=state, deterministic=True)
                #print("\naction: ", action)
                next_obs, rewards, done, _info = eval_env.step(action)
                print("rewards: ", rewards)
                #if not is_save: eval_env.render()
                eval_env.render()
                if not ever_done:
                        task_data.append([eval_env.venv.envs[0].world.state, action, rewards, done])
                        rets += rewards
                ever_done = np.logical_or(ever_done, done)
                obs = next_obs
                #if not is_save: time.sleep(.1)
                time.sleep(.1)
        if is_save:
                assert eval_dir
                with open(os.path.join(eval_dir, "task_data.pkl"), "wb") as f:
                        pickle.dump(task_data, f)
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




