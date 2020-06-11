import tensorflow as tf
from stable_baselines.ddpg.ddpg import normalize, tc, reduce
from stable_baselines.ddpg import DDPG
from stable_baselines.her import HER


class HER2L2SP(HER):
    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        data, _ = cls._load_from_file(load_path, custom_objects=custom_objects)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        data['model_class'] = DDPG2L2SP
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


class DDPG2L2SP(DDPG):

    def __init__(self, policy, env, **kwargs):
        super().__init__(policy, env, **kwargs)
        self.l2sp_coef = 0.01
        self.l2_coef = 0.0005

    def _setup_actor_optimizer(self):
        """
        setup the optimizer for the actor
        """
        if self.verbose >= 2:
            logger.info('setting up actor optimizer')

        ### L2SP LOSS ###
        all_vars = [v for v in tf.global_variables()]
        self.l2sp_loss = 0.0
        for var in all_vars:
            if 'pi' in var.name:
                self.l2sp_loss += tf.losses.mean_squared_error(self.original_params[var.name], var)
        self.l2_loss = 0.0
        for var in all_vars:
            if 'pi' in var.name:
                self.l2_loss += tf.losses.mean_squared_error(tf.zeros(var.shape), var)
        ### L2SP LOSS ###

        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf) + \
                          self.l2sp_coef * self.l2sp_loss + self.l2_coef * self.l2_loss
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

        ### L2SP LOSS ###
        all_vars = [v for v in tf.global_variables()]
        self.l2sp_loss = 0.0
        for var in all_vars:
            if 'qf' in var.name:
                self.l2sp_loss += tf.losses.mean_squared_error(self.original_params[var.name], var)
        self.l2_loss = 0.0
        for var in all_vars:
            if 'qf' in var.name:
                self.l2_loss += tf.losses.mean_squared_error(tf.zeros(var.shape), var)
        ### L2SP LOSS ###

        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms),
                                                       self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf)) + \
                           self.l2sp_coef * self.l2sp_loss + self.l2_coef * self.l2_loss
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
