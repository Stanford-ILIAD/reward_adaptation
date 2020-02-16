import os
import time
import csv
import numpy as np
import pickle
from scipy.special import kl_div

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
