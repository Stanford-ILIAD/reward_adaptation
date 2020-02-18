import os
import time
import csv
import numpy as np
import pickle
from scipy.special import kl_div
import tensorflow as tf


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


def print_param_names(model):
    """
    Prints model parameter names
    """
    for (param_name, param) in model.get_parameters().items():
        print(param_name, param.shape)

def get_param_idx(model, target_name):
    """
    Returns the index of the parameter in the list of params
    """
    for i, (param_name, param) in enumerate(model.get_parameters().items()):
        if param_name == target_name:
            return i

def add_random_noise(w, mean=0.0, stddev=1.0):
    variables_shape = tf.shape(w)
    noise = tf.random_normal(
        variables_shape,
        mean=mean,
        stddev=stddev,
        dtype=tf.float32,
    )
    return tf.assign_add(w, noise)

