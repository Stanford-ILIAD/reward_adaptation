import gym
import numpy as np
import time
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
import wandb
from tensorflow import flags
import driving_envs

FLAGS = flags.FLAGS
# PPO parameters
flags.DEFINE_integer("timesteps", 100000, "# timesteps to train")
# Experiment related parameters
flags.DEFINE_string("name", "test", "Name of experiment")
flags.DEFINE_string("logdir", "./test", "logdir")
flags.DEFINE_boolean("is_save", False, "Saves and logs experiment data if True")
flags.DEFINE_integer("eval_save_period", 10, "how often we save state for eval")


def train(timesteps, experiment_name, is_save, eval_save_period, log_dir, num_envs=1):
    env_fns = num_envs * [lambda: gym.make("Merging-v0")]
    env = VecNormalize(SubprocVecEnv(env_fns))
    eval_env = VecNormalize(DummyVecEnv(env_fns), training=False, norm_reward=False)
    policy = MlpPolicy
    model = PPO2(policy, env, verbose=1)

    def evaluate(model):
        """
        Evaluates model on one episode of driving task. Returns mean episode reward.
        """
        rets = 0.0
        obs = eval_env.reset()
        state, ever_done = None, False
        while not ever_done:
            action, state = model.predict(obs, state=state, deterministic=True)
            print("\n" + str(action))
            next_obs, rewards, done, _info = eval_env.step(action)
            eval_env.render()
            if not ever_done:
                rets += rewards
            ever_done = np.logical_or(ever_done, done)
            obs = next_obs
            time.sleep(.1)
        return np.mean(rets)

    n_steps = 0

    def callback(_locals, _globals):
        nonlocal n_steps
        model = _locals['self']
        if (n_steps + 1) % eval_save_period == 0:
            start_eval_time = time.time()
            mean_ret = evaluate(model)
            print("mean eval ret: ", mean_ret)
            end_eval_time = time.time() - start_eval_time
            print("Finished evaluation in {:.2f} seconds".format(end_eval_time))
        n_steps += 1
        return True

    model.learn(total_timesteps=timesteps, callback=callback)


if __name__ == '__main__':
    if FLAGS.is_save: wandb.init(project="reward_adaptation", sync_tensorboard=True)
    train(FLAGS.timesteps, FLAGS.name, FLAGS.is_save, FLAGS.eval_save_period, FLAGS.logdir)
