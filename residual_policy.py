from stable_baselines.common.policies import ActorCriticPolicy, nature_cnn, mlp_extractor, linear, CategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution, MultiCategoricalProbabilityDistribution

from stable_baselines.common import BaseRLModel
import json
import zipfile
from stable_baselines.common.save_util import data_to_json, json_to_data, params_to_bytes, bytes_to_params
import os
import numpy as np
import collections
import tensorflow as tf

def load_residual_model(cls, load_path, env=None, custom_objects=None, **kwargs):
        """
        Load the model from file
        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Envrionment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params = load_from_file(load_path, custom_objects=custom_objects)

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

def save_to_file(save_path, data=None, params=None):
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

def resave_model_residual(model_path):
    data, params = load_from_file(model_path)
    exec("data['policy'] = " + 'Residual'+str(data['policy']).split("'")[1].split('.')[-1])
    new_params = collections.OrderedDict()
    for key in params:
      new_params[key.replace('model', 'residual')] = params[key]
    save_to_file(os.path.join(os.path.dirname(model_path),'residual' + os.path.basename(model_path)), data, new_params)

class ResidualFeedForwardPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(ResidualFeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:
                pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(self.processed_obs), net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
        with tf.variable_scope("residual", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:
                pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(self.processed_obs), net_arch, act_fun)

            self._value_fn1 = linear(vf_latent, 'vf', 1)

            self._proba_distribution1, self._policy1, self.q_value1 = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)


        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


    def _setup_init(self):
        """Sets up the distributions, actions, and value."""
        with tf.variable_scope("output", reuse=True):
            assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None
            self._action = self.proba_distribution.sample()
            self._deterministic_action = self.proba_distribution.mode()
            self._neglogp = self.proba_distribution.neglogp(self.action)

            self._action += self._proba_distribution1.sample()
            self._deterministic_action += self._proba_distribution1.mode()
            
            if isinstance(self.proba_distribution, CategoricalProbabilityDistribution):
                self._policy_proba = tf.nn.softmax(self.policy)
            elif isinstance(self.proba_distribution, DiagGaussianProbabilityDistribution):
                self._policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]
            elif isinstance(self.proba_distribution, BernoulliProbabilityDistribution):
                self._policy_proba = tf.nn.sigmoid(self.policy)
            elif isinstance(self.proba_distribution, MultiCategoricalProbabilityDistribution):
                self._policy_proba = [tf.nn.softmax(categorical.flatparam())
                                     for categorical in self.proba_distribution.categoricals]
            else:
                self._policy_proba = []  # it will return nothing, as it is not implemented
            self._value_flat = self.value_fn[:, 0]

class ResidualMlpPolicy(ResidualFeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(ResidualMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)

if __name__ == '__main__':
  data, params = load_from_file('/mnt/parallel/caozj/reward_adaptation/reward_curriculum_expts/weight_-1-100-residual/best_model_107520_[-1212.9834].pkl')
  data1, params1 = load_from_file('/mnt/parallel/caozj/reward_adaptation/reward_curriculum_expts/weight_-1/best_model_151040_[710.741].pkl')
  import pdb
  pdb.set_trace()
