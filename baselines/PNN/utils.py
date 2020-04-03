import os
import glob
import json
import zipfile
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from typing import Union, List, Callable, Optional

import gym
import cloudpickle
import numpy as np
import tensorflow as tf

from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.save_util import data_to_json, json_to_data, params_to_bytes, bytes_to_params
from stable_baselines import logger

from stable_baselines.common import BaseRLModel

from model import MlpPPNPolicy


def load_from_file(load_path, load_data=True, custom_objects=None):
    """Load model data from a .zip archive
    :param load_path: (str or file-like) Where to load model from
    :param load_data: (bool) Whether we should load and return data
        (class parameters). Mainly used by `load_parameters` to
        only load model parameters (weights).
    :param custom_objects: (dict) Dictionary of objects to replace
        upon loading. If a variable is present in this dictionary as a
        key, it will not be deserialized and the corresponding item
        will be used instead. Similar to custom_objects in
        `keras.models.load_model`. Useful when you have an object in
        file that can not be deserialized.
    :return: (dict, OrderedDict) Class parameters and model parameters
    """
    # Check if file exists if load_path is
    # a string
    if isinstance(load_path, str):
        if not os.path.exists(load_path):
            if os.path.exists(load_path + ".zip"):
                load_path += ".zip"
            else:
                raise ValueError("Error: the file {} could not be found".format(load_path))

    # Open the zip archive and load data.
    try:
        with zipfile.ZipFile(load_path, "r") as file_:
            namelist = file_.namelist()
            # If data or parameters is not in the
            # zip archive, assume they were stored
            # as None (_save_to_file allows this).
            data = None
            params = None
            if "data" in namelist and load_data:
                # Load class parameters and convert to string
                # (Required for json library in Python 3.5)
                json_data = file_.read("data").decode()
                data = json_to_data(json_data, custom_objects=custom_objects)

            if "parameters" in namelist:
                # Load parameter list and and parameters
                parameter_list_json = file_.read("parameter_list").decode()
                parameter_list = json.loads(parameter_list_json)
                serialized_params = file_.read("parameters")
                params = bytes_to_params(
                    serialized_params, parameter_list
                )
    except zipfile.BadZipFile:
        # load_path wasn't a zip file. Possibly a cloudpickle
        # file. Show a warning and fall back to loading cloudpickle.
        warnings.warn("It appears you are loading from a file with old format. " +
                      "Older cloudpickle format has been replaced with zip-archived " +
                      "models. Consider saving the model with new format.",
                      DeprecationWarning)
        # Attempt loading with the cloudpickle format.
        # If load_path is file-like, seek back to beginning of file
        if not isinstance(load_path, str):
            load_path.seek(0)
        data, params = BaseRLModel._load_from_file_cloudpickle(load_path)

    return data, params

def save_to_file(save_path, data=None, params=None, cloudpickle=False):
    """Save model to a zip archive or cloudpickle file.
    :param save_path: (str or file-like) Where to store the model
    :param data: (OrderedDict) Class parameters being stored
    :param params: (OrderedDict) Model parameters being stored
    :param cloudpickle: (bool) Use old cloudpickle format
        (stable-baselines<=2.7.0) instead of a zip archive.
    """
    if cloudpickle:
        save_to_file_cloudpickle(save_path, data, params)
    else:
        save_to_file_zip(save_path, data, params)

def save_to_file_cloudpickle(save_path, data=None, params=None):
        """Legacy code for saving models with cloudpickle
        :param save_path: (str or file-like) Where to store the model
        :param data: (OrderedDict) Class parameters being stored
        :param params: (OrderedDict) Model parameters being stored
        """
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                save_path += ".pkl"

            with open(save_path, "wb") as file_:
                cloudpickle.dump((data, params), file_)
        else:
            # Here save_path is a file-like object, not a path
            cloudpickle.dump((data, params), save_path)

def save_to_file_zip(save_path, data=None, params=None):
        """Save model to a .zip archive
        :param save_path: (str or file-like) Where to store the model
        :param data: (OrderedDict) Class parameters being stored
        :param params: (OrderedDict) Model parameters being stored
        """
        # data/params can be None, so do not
        # try to serialize them blindly
        if data is not None:
            serialized_data = data_to_json(data)
        if params is not None:
            serialized_params = params_to_bytes(params)
            # We also have to store list of the parameters
            # to store the ordering for OrderedDict.
            # We can trust these to be strings as they
            # are taken from the Tensorflow graph.
            serialized_param_list = json.dumps(
                list(params.keys()),
                indent=4
            )

        # Check postfix if save_path is a string
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                save_path += ".zip"

        # Create a zip-archive and write our objects
        # there. This works when save_path
        # is either str or a file-like
        with zipfile.ZipFile(save_path, "w") as file_:
            # Do not try to save "None" elements
            if data is not None:
                file_.writestr("data", serialized_data)
            if params is not None:
                file_.writestr("parameters", serialized_params)
                file_.writestr("parameter_list", serialized_param_list)

def looseload(cls, load_path, env=None, custom_objects=None, **kwargs):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Environment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params = cls._load_from_file(load_path, custom_objects=custom_objects)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=None, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        model.load_parameters(params, exact_match=False)

        return model

def resave_params_for_PPN(input_file, output_file):
    data, params = load_from_file(input_file)
    data['policy'] = MlpPPNPolicy
    remove_keys = ['model/vf/w:0', 'model/vf/b:0', 'model/pi/w:0', 'model/pi/b:0', 'model/pi/logstd:0', 'model/q/w:0', 'model/q/b:0']
    for remove_key in remove_keys:
        params.pop(remove_key)
    save_to_file(output_file, data, params)


if __name__ == '__main__':
    data, params = load_from_file('/iliad/u/caozj/reward_adaptation/output/gridworld_continuous/MO_LL2RL_backward_step_13/model_256_2330.0870387565815.pkl')
    data['policy'] = MlpPPNPolicy
    remove_keys = ['model/vf/w:0', 'model/vf/b:0', 'model/pi/w:0', 'model/pi/b:0', 'model/pi/logstd:0', 'model/q/w:0', 'model/q/b:0']
    for remove_key in remove_keys:
        params.pop(remove_key)
    save_to_file('test_model.pkl', data, params)
    import pdb
    pdb.set_trace()
