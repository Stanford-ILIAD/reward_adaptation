import warnings
from itertools import zip_longest
import numpy as np
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.ddpg.policies import DDPGPolicy

def nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


class PNNPolicy(DDPGPolicy):
    """
    Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn",
                 layer_norm=False, act_fun=tf.nn.relu, **kwargs):
        super(PNNPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = cnn_extractor
        self.reuse = reuse
        self._qvalue = None
        if layers is None:
            layers = [64, 64]
        self.layers = layers

        assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

        self.activ = act_fun

    def make_actor_old(self, obs=None, reuse=False, scope="pi"):
        latent_pi = []
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(obs)
            for i, layer_size in enumerate(self.layers):
                pi_h = tf.layers.dense(pi_h, layer_size, name='fc' + str(i))
                if self.layer_norm:
                    pi_h = tf.contrib.layers.layer_norm(pi_h, center=True, scale=True)
                pi_h = self.activ(pi_h)
                latent_pi.append(pi_h)
        return pi_h, latent_pi

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        pi_h, latent_pi = self.make_actor_old(obs=obs, reuse=reuse, scope=scope)
        all_latents = []
        for latent in latent_pi:
            all_latents.append([latent])
        with tf.variable_scope(scope+"1", reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(obs)
            for i, layer_size in enumerate(self.layers):
                pi_h = tf.layers.dense(pi_h, layer_size, name='fc' + str(i))
                if self.layer_norm:
                    pi_h = tf.contrib.layers.layer_norm(pi_h, center=True, scale=True)
                pi_h = self.activ(pi_h) + sum(all_latents[i])
            self.policy = tf.nn.tanh(tf.layers.dense(pi_h, self.ac_space.shape[0], name=scope,
                                                     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                                      maxval=3e-3)))
        return self.policy

    def make_critic_old(self, obs=None, action=None, reuse=False, scope="qf"):
        latent_qf = []
        if obs is None:
            obs = self.processed_obs
        if action is None:
            action = self.action_ph

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                qf_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                qf_h = tf.layers.flatten(obs)
            for i, layer_size in enumerate(self.layers):
                qf_h = tf.layers.dense(qf_h, layer_size, name='fc' + str(i))
                if self.layer_norm:
                    qf_h = tf.contrib.layers.layer_norm(qf_h, center=True, scale=True)
                qf_h = self.activ(qf_h)
                if i == 0:
                    qf_h = tf.concat([qf_h, action], axis=-1)
                latent_qf.append(qf_h)

        return qf_h, latent_qf

    def make_critic(self, obs=None, action=None, reuse=False, scope="qf"):
        if obs is None:
            obs = self.processed_obs
        if action is None:
            action = self.action_ph

        qf_h, latent_qf = self.make_critic_old(obs=obs, action=action, reuse=reuse, scope=scope)
        all_latents = []
        for latent in latent_qf:
            all_latents.append([latent])

        with tf.variable_scope(scope+"1", reuse=reuse):
            if self.feature_extraction == "cnn":
                qf_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                qf_h = tf.layers.flatten(obs)
            for i, layer_size in enumerate(self.layers):
                qf_h = tf.layers.dense(qf_h, layer_size, name='fc' + str(i))
                if self.layer_norm:
                    qf_h = tf.contrib.layers.layer_norm(qf_h, center=True, scale=True)
                qf_h = self.activ(qf_h)
                if i == 0:
                    qf_h = tf.concat([qf_h, action], axis=-1)
                    qf_h += sum(all_latents[i])
                else:
                    qf_h = self.activ(qf_h) + sum(all_latents[i])

            # the name attribute is used in pop-art normalization
            qvalue_fn = tf.layers.dense(qf_h, 1, name='qf_output',
                                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                         maxval=3e-3))

            self.qvalue_fn = qvalue_fn
            self._qvalue = qvalue_fn[:, 0]
        return self.qvalue_fn

    def step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def value(self, obs, action, state=None, mask=None):
        return self.sess.run(self._qvalue, {self.obs_ph: obs, self.action_ph: action})


class MlpPPNPolicy(PNNPolicy):
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
        super(MlpPPNPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)


import functools
from stable_baselines.ddpg.ddpg import normalize, tc, reduce
from stable_baselines.ddpg import DDPG
from stable_baselines.her import HER

import ipdb

class HER2PNN(HER):
    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        data, _ = cls._load_from_file(load_path, custom_objects=custom_objects)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        data['model_class'] = DDPG2PNN
        print("data policy", data['policy'])
        model = cls(policy=data["policy"], env=env, model_class=data['model_class'],
                    n_sampled_goal=data['n_sampled_goal'],
                    goal_selection_strategy=data['goal_selection_strategy'],
                    _init_setup_model=False)
        model.__dict__['observation_space'] = data['her_obs_space']
        model.__dict__['action_space'] = data['her_action_space']
        model.model = data['model_class'].load(load_path, model.get_env(), **kwargs)
        model.model._save_to_file = model._save_to_file
        return model


from stable_baselines import logger
from stable_baselines.common import tf_util
from stable_baselines.common.mpi_adam import MpiAdam

class DDPG2PNN(DDPG):

    def __init__(self, policy, env, **kwargs):
        super().__init__(policy, env, **kwargs)
        self.bss_coef = 0.001
        self.l2_coef = 0.0005
        print(self.bss_coef, self.l2_coef)

    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        """
        Same as DDPG code, excpet loading parameters without exact match
        :param load_path:
        :param env:
        :param custom_objects:
        :param kwargs:
        :return:
        """
        data, params = cls._load_from_file(load_path, custom_objects=custom_objects)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(None, env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()
        # Patch for version < v2.6.0, duplicated keys where saved
        if len(params) > len(model.get_parameter_list()):
            n_params = len(model.params)
            n_target_params = len(model.target_params)
            n_normalisation_params = len(model.obs_rms_params) + len(model.ret_rms_params)
            # Check that the issue is the one from
            # https://github.com/hill-a/stable-baselines/issues/363
            assert len(params) == 2 * (n_params + n_target_params) + n_normalisation_params, \
                "The number of parameter saved differs from the number of parameters" \
                " that should be loaded: {}!={}".format(len(params), len(model.get_parameter_list()))
            # Remove duplicates
            params_ = params[:n_params + n_target_params]
            if n_normalisation_params > 0:
                params_ += params[-n_normalisation_params:]
            params = params_
        print("loading: exact match FALSE")
        model.load_parameters(params, exact_match=False)

        return model

    def _setup_actor_optimizer(self):
        """
        setup the optimizer for the actor
        """
        if self.verbose >= 2:
            logger.info('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list() for var in tf_util.get_trainable_vars('model/pi/')]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        if self.verbose >= 2:
            logger.info('  actor shapes: {}'.format(actor_shapes))
            logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = tf_util.flatgrad(self.actor_loss, tf_util.get_trainable_vars('model/pi/'),
                                            clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('model/pi/'), beta1=0.9, beta2=0.999,
                                       epsilon=1e-08)

    def _setup_critic_optimizer(self):
        """
        setup the optimizer for the critic
        """
        if self.verbose >= 2:
            logger.info('setting up critic optimizer')

        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms),
                                                       self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in tf_util.get_trainable_vars('model/qf/')
                               if 'bias' not in var.name and 'qf_output' not in var.name and 'b' not in var.name]
            if self.verbose >= 2:
                for var in critic_reg_vars:
                    logger.info('  regularizing: {}'.format(var.name))
                logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in tf_util.get_trainable_vars('model/qf/')]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        if self.verbose >= 2:
            logger.info('  critic shapes: {}'.format(critic_shapes))
            logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = tf_util.flatgrad(self.critic_loss, tf_util.get_trainable_vars('model/qf/'),
                                             clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('model/qf/'), beta1=0.9, beta2=0.999,
                                        epsilon=1e-08)

