import numpy as np
import time, os
import torch
from torch.optim import Adam
import gym
import toy.core as core
#from toy.logx import EpochLogger
from toy.logger import EpochLogger
from toy.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from toy.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import driving.driving_envs
import torch.nn as nn
import ipdb
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from toy.utils import *
from mpl_toolkits.mplot3d import Axes3D
os.environ['KMP_DUPLICATE_LIB_OK']='True'



class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



class VPG():
    def __init__(self, env_fn, exp_name, actor_critic=core.MLPActorCritic, ac_kwargs=dict(),  seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1, save=False):

        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_v_iters = train_v_iters
        self.epochs = epochs
        self.save_freq = save_freq
        self.steps_per_epoch = steps_per_epoch
        self.save = save
        self.output_dir = "toy/output/" + exp_name

        # random seed
        self.seed = seed + 10000 * proc_id()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


        # instantiate environment
        self.env = env_fn()
        self.eval_env = env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.max_ep_len = self.env.time_limit

        # Exploration
        self.exploration = 0.5
        self.init_exp = 0.5
        self.final_exp = 0.0
        self.anneal_steps = 4000
        self.train_iteration = 0

        # create actor-critic module
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, self.exploration, **ac_kwargs)

        # set up experience buffer
        self.local_steps_per_epoch = int(steps_per_epoch/num_procs())
        self.buf = VPGBuffer(self.obs_dim, self.act_dim, self.local_steps_per_epoch, gamma, lam)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

        # Set up model saving and logging
        if self.save:
            self.logger = EpochLogger(**logger_kwargs)
            self.logger.save_config(locals())
            self.logger.setup_pytorch_saver(self.ac)
            # Count variables
            #var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.v])
            #self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)


    def _compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # policy loss
        pi, logp = self.ac.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        # useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    def _compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def _update(self, epoch, train=True):
        data = self.buf.get()

        # Get loss and info values before update
        pi_l_old, pi_info_old = self._compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self._compute_loss_v(data).item()

        if not train:
            return pi_l_old

        # Save weights and loss
        curr_weight = self.ac.pi.mu_net[0].weight[0]
        curr_std = self.ac.pi.log_std[0]
        if self.save:
            self.logger.writerow([epoch, curr_weight[0].item(), curr_weight[1].item(), curr_std.item(), pi_l_old])

        # Train policy with a single step of gradient descent
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self._compute_loss_pi(data)
        loss_pi.backward()
        mpi_avg_grads(self.ac.pi)  # average grads across MPI processes
        self.pi_optimizer.step()

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self._compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.ac.v)  # average grads across MPI processes
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        if self.save:
            self.logger.store(LossPi=pi_l_old, LossV=v_l_old,
                         KL=kl, Entropy=ent,
                         DeltaLossPi=(loss_pi.item() - pi_l_old),
                         DeltaLossV=(loss_v.item() - v_l_old))

        if self.save:
            self.logger.scalar_summary("loss_pi", loss_pi.item(), epoch + 1)
            self.logger.scalar_summary("loss_v", loss_v.item(), epoch + 1)
            for tag, value in self.ac.pi.named_parameters():
                self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
                if value.grad != None:
                    self.logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

        self._anneal_exploration()

        return loss_pi

    def _anneal_exploration(self):
        ratio = max((self.anneal_steps - self.train_iteration) / float(self.anneal_steps), 0)
        self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp

    def _eval(self, epoch):
        #self.eval_logger = EpochLogger()
        num_episodes = 1
        render = False if self.save else True
        o, r, d, ep_ret, ep_len, n = self.eval_env.reset(), 0, False, 0, 0, 0
        while n < num_episodes:
            if render:
                self.eval_env.render()
                time.sleep(1e-3)

            a_unclipped = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
            a = np.clip(a_unclipped, self.env.action_space.low, self.env.action_space.high)
            #print("a: ", a_unclipped, a)
            o, r, d, _ = self.eval_env.step(a, verbose=False)
            ep_ret += r
            ep_len += 1

            if d or (ep_len == self.max_ep_len):
                #self.eval_logger.store(EpRet=ep_ret, EpLen=ep_len)
                if self.save: self.logger.scalar_summary("eval_ret", ep_ret, epoch + 1)
                print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
                if ep_ret >0:
                    if self.save: self.logger.save_state({'env': self.env}, epoch, ep_ret)
                o, r, d, ep_ret, ep_len = self.eval_env.reset(), 0, False, 0, 0
                n += 1

        #if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
        #self.eval_logger.log_tabular('EpRet', with_min_and_max=True)
        #self.eval_logger.log_tabular('EpLen', average_only=True)
        #self.eval_logger.dump_tabular()

    def train(self):
        # Prepare for interaction with environment
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.epochs):
            print("EPOCH {} WEIGHT {}".format(epoch, self.ac.pi.mu_net[0].weight[0]))
            print("exploration: ", self.exploration)
            for t in range(self.local_steps_per_epoch):
                #self.env.render()
                #ipdb.set_trace()
                a_unclipped, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                a = np.clip(a_unclipped, self.env.action_space.low, self.env.action_space.high)
                #print("unclipped: ", a_unclipped, "clipped: ", a)

                next_o, r, d, _ = self.env.step(a, verbose=False)
                ep_ret += r
                ep_len += 1

                # save and log
                self.buf.store(o, a, r, v, logp)
                if self.save: self.logger.store(VVals=v)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.local_steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not (terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                    self.buf.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        if self.save: self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    print("EP RET: ", ep_ret, "Len: ", ep_len)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0
                self.train_iteration += 1

            # Save model
            #if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
            #    self.logger.save_state({'env': self.env}, None)

            # Perform VPG update!
            self._update(epoch)

            # Log info about epoch
            if self.save:
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('VVals', with_min_and_max=True)
                self.logger.log_tabular('TotalEnvInteracts', (epoch + 1) * self.steps_per_epoch)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossV', average_only=True)
                self.logger.log_tabular('DeltaLossPi', average_only=True)
                self.logger.log_tabular('DeltaLossV', average_only=True)
                self.logger.log_tabular('Entropy', average_only=True)
                self.logger.log_tabular('KL', average_only=True)
                self.logger.log_tabular('Time', time.time() - start_time)
                self.logger.dump_tabular()

            # Evaluate model
            self._eval(epoch)

        self.get_mesh_grid()
        ipdb.set_trace()

    def fine_tune(self, source_model_file):
        print("loading source model params..")
        self.ac = torch.load(source_model_file)
        print("fine-tuning")
        self.train()

    def get_loss_value(self, weight):
        self.ac.pi.mu_net[0].weight[0] = torch.Tensor(weight)
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        for t in range(self.local_steps_per_epoch):
            #self.env.render()
            a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32), is_exp=False)
            a = np.clip(a, self.env.action_space.low, self.env.action_space.high)
            # print("action alter: ", a)

            next_o, r, d, _ = self.env.step(a, verbose=False)
            ep_ret += r
            ep_len += 1

            # save
            self.buf.store(o, a, r, v, logp)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == self.max_ep_len
            terminal = d or timeout
            epoch_ended = t == self.local_steps_per_epoch - 1

            if terminal or epoch_ended:
                #if epoch_ended and not (terminal):
                #    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32), is_exp=False)
                else:
                    v = 0
                self.buf.finish_path(v)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

        # Perform VPG update!
        loss_pi = self._update(epoch=None, train=False)
        return loss_pi

    def get_mesh_grid(self):
        #ipdb.set_trace()
        X, Y = np.meshgrid(np.arange(-1,1,0.1), np.arange(-1,1,0.1))
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            print("row: ", i)
            for j in range(X.shape[1]):
                x, y = X[i][j], Y[i][j]
                weight = np.array([x,y])
                loss_pi = self.get_loss_value(weight)
                Z[i][j] = loss_pi

        loss_plot = {'X': X, 'Y': Y, 'Z': Z}
        pickle.dump(loss_plot, open(self.output_dir+"/grid_{}_{}_{}.pkl".format(-1, 1, 0.1), "wb"))
        return X, Y, Z


    def plot(self, x, y, z):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, y, z,
                        cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=True)
        wx,wy,wz = get_weights_from_csv(self.output_dir+"/weight_loss.csv")
        ax.scatter3D(wx, wy, wz, c="black", s=5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        ipdb.set_trace()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Toy-v0')
    parser.add_argument('--hid', type=int, default=1)
    parser.add_argument('--l', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--pi_lr', type=float, default=1e-3)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--save', action='store_true')

    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--epret', type=float, default=1.0)
    parser.add_argument('--fpath', type=str)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from toy.run_utils import setup_logger_kwargs
    #logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger_kwargs = {"output_dir": "toy/output", "exp_name": args.exp_name}

    gym.register(
        id=args.env,
        entry_point="driving_envs.envs:GridworldToyEnv",
    )

    vpg = VPG(lambda : gym.make(args.env), args.exp_name, actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, activation=nn.Identity),
        gamma=args.gamma, pi_lr=args.pi_lr, seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, save_freq=args.save_freq, save=args.save)

    # Train
    #vpg.train()

    # Fine-Tune
    #import os.path as osp
    #fname = osp.join(args.fpath, 'pyt_save', 'model' + str(args.itr) + '_' + str(args.epret) + '.pt')
    #print("args exp name", args.exp_name)
    #print('\n\nLoading from %s\n\n' % fname)
    #vpg.fine_tune(fname)

    # Get mesh grid
    #x,y,z = vpg.get_mesh_grid()

    # Plot
    print("plotting")
    with open("toy/output/"+args.exp_name+"/grid_{}_{}_{}.pkl".format(-1,1,0.1), "rb") as f:
        loss_plot_dict = pickle.load(f)
    x,y,z = loss_plot_dict['X'], loss_plot_dict['Y'], loss_plot_dict['Z']
    vpg.plot(x,y,z)






