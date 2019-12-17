import os
import time
import csv
import numpy as np
import pickle
from stable_baselines.common.distributions import DiagGaussianProbabilityDistribution as DGPD
from scipy.special import kl_div


#def evaluate_debug(model, eval_env):
#    """
#        Evaluates model on one episode of driving task. Returns mean episode reward.
#        """
#    rets = 0.0
#    obs = eval_env.reset()
#    state, ever_done = None, False
#    task_data = []
#    while not ever_done:
#        action, state = model.predict(obs, state=state, deterministic=True)
#        next_obs, rewards, done, _info = eval_env.step(action)
#        # eval_env.render()
#        if not ever_done:
#            task_data.append([eval_env.venv.envs[0].world.state, action, rewards, done])
#            rets += rewards
#        ever_done = np.logical_or(ever_done, done)
#        obs = next_obs
#        # time.sleep(.1)
#    return rets

def evaluate_action(m1, m2, eval_env, eval_dir=None):
    """
    Evaluates model on one episode of driving task. Returns mean episode reward.
    """

    total_rets = []
    KL_divs = []
    for e in range(10):
        rets = 0.0
        obs = eval_env.reset()
        state, done = None, False
        while not done:
            action, state = m1.predict(obs, state=state, deterministic=True)

            # calculate KL
            m1_ap = m1.action_probability(obs, state=state, actions=action)[0]
            m2_ap = m2.action_probability(obs, state=state, actions=action)[0]
            KL_div = kl_div(m1_ap, m2_ap)
            print("p, q: ", m1_ap, m2_ap, KL_div, m1_ap >= m2_ap)
            if KL_div == np.infty:
                print(KL_div, m1_ap, m2_ap)
            KL_divs.append(KL_div)

            next_obs, rewards, done, _info = eval_env.step(action)
            if not done:
                rets += rewards
            obs = next_obs
        total_rets.append(rets[0])
    return np.mean(KL_divs)


def evaluate_debug(model, eval_env):
        """
        Evaluates model on 10 episodes of driving task.
        Returns mean episode reward and standard deviation.
        """
        total_rets = []
        for e in range(10):
            rets = 0.0
            obs = eval_env.reset()
            state, done = None, False
            # task_data = []
            while not done:
                action, state = model.predict(obs, state=state, deterministic=True)
                # print("\naction", action)
                next_obs, rewards, done, _info = eval_env.step(action)
                eval_env.render()
                if not done:
                    # task_data.append([eval_env.venv.envs[0].world.state, action, rewards, done])
                    rets += rewards
                # ever_done = np.logical_or(ever_done, done)
                obs = next_obs
                time.sleep(.1)
            total_rets.append(rets[0])
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
