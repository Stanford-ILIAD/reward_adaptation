import warnings
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines.common.policies import nature_cnn, RecurrentActorCriticPolicy, ActorCriticPolicy, mlp_extractor
from stable_baselines.ddpg.policies import DDPGPolicy
import ipdb



class BSSPolicy(DDPGPolicy):
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
        super(BSSPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
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

    def make_actor(self, obs=None, reuse=False, scope="pi"):
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
            self.policy = tf.nn.tanh(tf.layers.dense(pi_h, self.ac_space.shape[0], name=scope,
                                                     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                                      maxval=3e-3)))
            self.pi_feature_m = pi_h
        return self.policy

    def make_critic(self, obs=None, action=None, reuse=False, scope="qf"):
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

            # the name attribute is used in pop-art normalization
            qvalue_fn = tf.layers.dense(qf_h, 1, name='qf_output',
                                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                         maxval=3e-3))

            self.qvalue_fn = qvalue_fn
            self._qvalue = qvalue_fn[:, 0]
            self.qf_feature_m = qf_h
        return self.qvalue_fn

    def feature_matrices(self):
        return self.pi_feature_m, self.qf_feature_m

    def step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def value(self, obs, action, state=None, mask=None):
        return self.sess.run(self._qvalue, {self.obs_ph: obs, self.action_ph: action})


class BSSPolicy_ActorCritic(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(BSSPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
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

            self.pi_feature_m = pi_latent
            self.vf_feature_m = vf_latent

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def feature_matrices(self):
        return self.pi_feature_m, self.vf_feature_m

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


class MlpBSSPolicy(BSSPolicy):
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
        super(MlpBSSPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)




from stable_baselines.ddpg.ddpg import normalize, tc, reduce, MpiAdam, denormalize, reduce_std, \
    reduce_var, get_target_updates, get_perturbed_actor_updates
from stable_baselines.ddpg import DDPG
from stable_baselines.her import HER


class HER2BSS(HER):
    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        data, _ = cls._load_from_file(load_path, custom_objects=custom_objects)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        data['model_class'] = DDPG2BSS
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
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.mpi_adam import MpiAdam


class DDPG2BSS(DDPG):

    def __init__(self, policy, env, **kwargs):
        super().__init__(policy, env, **kwargs)
        self.bss_coef = 0.001
        self.l2_coef = 0.0005
        print(self.bss_coef, self.l2_coef)

    def _setup_actor_optimizer(self):
        """
        setup the optimizer for the actor
        """
        if self.verbose >= 2:
            logger.info('setting up actor optimizer')

        ### BSS LOSS ###
        all_vars = [v for v in tf.global_variables()]
        self.l2_loss = 0.0
        for var in all_vars:
            if 'pi' in var.name:
                self.l2_loss += tf.losses.mean_squared_error(tf.zeros(var.shape), var)

        pi_features, _ = self.policy_tf.feature_matrices()
        singular_pi = tf.linalg.svd(pi_features, compute_uv=False)
        self.bss_loss = tf.reduce_sum(tf.square(singular_pi[-1]))
        ### BSS LOSS ###

        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf) + \
                          self.bss_coef * self.bss_loss + self.l2_coef * self.l2_loss
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

        ### BSS LOSS ###
        all_vars = [v for v in tf.global_variables()]
        self.l2_loss = 0.0
        for var in all_vars:
            if 'qf' in var.name:
                self.l2_loss += tf.losses.mean_squared_error(tf.zeros(var.shape), var)

        _, qf_features = self.policy_tf.feature_matrices()
        singular_qf = tf.linalg.svd(qf_features, compute_uv=False)
        self.bss_loss = tf.reduce_sum(tf.square(singular_qf[-1]))
        ### BSS LOSS ###

        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms),
                                                       self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf)) + \
            self.bss_coef * self.bss_loss + self.l2_coef * self.l2_loss
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


