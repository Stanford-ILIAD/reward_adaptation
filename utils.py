import os
import time
import csv
import numpy as np
import pickle
from stable_baselines.common.distributions import DiagGaussianProbabilityDistribution as DGPD
from scipy.special import kl_div


weight_n1 = ("driving_policies", "weight_-1", "best_model_151040_[710.741].pkl")
weight_n05 = ("driving_policies", "weight_-0.5", "best_model_1428480_[309.0573].pkl")
weight_0 = ("driving_policies", "weight_0", "best_model_87040_[-37.172234].pkl")
weight_p05 = ("driving_policies", "weight_0.5", "best_model_16640_[-33.3783].pkl")
weight_p1 = ("driving_policies", "weight_1", "best_model_8960_[-29.448769].pkl")
weight_p2 = ("driving_policies", "weight_2", "best_model_1980160_[1087.1274].pkl")
weight_p4 = ("driving_policies", "weight_4", "best_model_1827840_[2262.8826].pkl")
weight_p6 = ("driving_policies", "weight_6", "best_model_2670080_[3432.1975].pkl")
weight_p8 = ("driving_policies", "weight_8", "best_model_3175680_[4600.899].pkl")
weight_p10 = ("driving_policies", "weight_10", "best_model_3527680_[5769.4253].pkl")
weight_p100 = ("driving_policies", "weight_100", "best_model_14080_[58647.805].pkl")

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
        for e in range(1):
            rets = 0.0
            obs = eval_env.reset()
            state, done = None, False
            while not done:
                action, state = model.predict(obs, state=state, deterministic=True)
                next_obs, ret, done, _info = eval_env.step(action)
                eval_env.render()
                if not done:
                    rets += ret
                obs = next_obs
                time.sleep(.1)
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
