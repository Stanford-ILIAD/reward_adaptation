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
        #self.buf = VPGBuffer(self.obs_dim, self.act_dim, self.local_steps_per_epoch, gamma, lam)

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

    def _compute_loss_pi(self, obs, act, weights, old_logps):
        # policy loss
        pi, logp = self.ac.pi(obs, act)
        loss_pi = -(logp * weights).mean()

        # useful extra info
        approx_kl = (old_logps - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)
        return loss_pi, pi_info

    def _update(self, obs, act, weights, old_logps, epoch):

        # Get loss and info values before update
        pi_l_old, pi_info_old = self._compute_loss_pi(obs, act, weights, old_logps)
        pi_l_old = pi_l_old.item()

        # Save initial weights and loss
        if self.save and epoch == 0:  # log initial loss as well
            curr_weight = self.ac.pi.mu_net[0].weight[0]
            curr_std = self.ac.pi.log_std[0]
            self.logger.writerow([epoch, curr_weight[0].item(), curr_weight[1].item(), curr_std.item(), pi_l_old])

        # Train policy with a single step of gradient descent
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self._compute_loss_pi(obs, act, weights, old_logps)

        loss_pi.backward()
        mpi_avg_grads(self.ac.pi)  # average grads across MPI processes
        self.pi_optimizer.step()

        # Save weights and loss
        if self.save and epoch%100==0:
            curr_weight = self.ac.pi.mu_net[0].weight[0]
            curr_std = self.ac.pi.log_std[0]
            self.logger.writerow([epoch+1, curr_weight[0].item(), curr_weight[1].item(), curr_std.item(), loss_pi.item()])

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        if self.save:
            self.logger.store(LossPi=pi_l_old, LossV=0,
                              KL=kl, Entropy=ent,
                              DeltaLossPi=(loss_pi.item() - pi_l_old),)

        if self.save:
            self.logger.scalar_summary("loss_pi", loss_pi.item(), epoch+1)
            for tag, value in self.ac.pi.named_parameters():
                self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
                if value.grad != None:
                    self.logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)
        return loss_pi


    def _normalize_obs(self, obs):
        # between -1 and 1
        mean_obs = 25 # max inp: 50, min inp: 0
        obs = (obs - mean_obs)/mean_obs

        # between 0 and 1
        #max_obs = 50
        #obs /= max_obs

        obs *= (self.env.action_space.high)
        #assert (obs <= 0.08).all() and (obs >= -0.01).all()
        return obs

    def _train_loop(self):
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []
        batch_old_logps = []
        o, d, ep_rews = self.env.reset(), False, []
        while True:
            self.env.render()
            # save obs
            o = self._normalize_obs(o)
            print("normalized o: ", o)
            batch_obs.append(o.copy())

            # act
            a_unclipped, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
            a = np.clip(a_unclipped, self.env.action_space.low, self.env.action_space.high)
            o, r, d, _ = self.env.step(a, verbose=True)

            # save action, reward
            print("a: ,", a_unclipped,a)
            batch_acts.append(a)
            batch_old_logps.append(logp.item())
            ep_rews.append(r)
            if self.save: self.logger.store(VVals=v)

            if d:
                # if epsiode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                o, d, ep_rews = self.env.reset(), False, []

                # log episode length and return
                if self.save: self.logger.store(EpRet=ep_ret, EpLen=ep_len)

                # end experience loop if we have enough of it
                if len(batch_obs) > self.local_steps_per_epoch:
                    return batch_obs, batch_acts, batch_weights, batch_old_logps


    def train(self):
        curr_weight = self.ac.pi.mu_net[0].weight[0]
        curr_std = self.ac.pi.log_std[0]
        print("curr weight, std: ", curr_weight, curr_std)
        start_time = time.time()
        for epoch in range(self.epochs):
            print("epoch: ", epoch)
            batch_obs, batch_acts, batch_weights, batch_old_logps = self._train_loop()
            loss_pi = (self._update(
                         obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                         act=torch.as_tensor(batch_acts, dtype=torch.float32),
                         weights=torch.as_tensor(batch_weights, dtype=torch.float32),
                         old_logps=torch.as_tensor(batch_old_logps, dtype=torch.float32),
                         epoch=epoch))
            # Log info about epoch
            if self.save:
                self.logger.log_tabular('Epoch', epoch+1)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('TotalEnvInteracts', (epoch + 1) * self.steps_per_epoch)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('DeltaLossPi', average_only=True)
                self.logger.log_tabular('Entropy', average_only=True)
                self.logger.log_tabular('KL', average_only=True)
                self.logger.log_tabular('Time', time.time() - start_time)
                self.logger.dump_tabular()

        ipdb.set_trace()

    def fine_tune(self, source_model_file):
        print("loading source model params..")
        #self.ac = torch.load(source_model_file)
        with torch.no_grad():
            self.ac.pi.mu_net[0].weight[0] = torch.Tensor(np.array([-0.75,0.75]))
        print("fine-tuning")
        self.train()

    def get_loss_value(self, weight):
        self.ac.pi.mu_net[0].weight[0] = torch.Tensor(weight)
        curr_weight = self.ac.pi.mu_net[0].weight[0]
        curr_std = self.ac.pi.log_std[0]
        loss_pis = []
        for rep in range(10):  # take samples of loss value because it is dependent on policy
            batch_obs, batch_acts, batch_weights, batch_old_logps = self._train_loop()
            loss_pi, pi_info = self._compute_loss_pi(
                                   obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                   act=torch.as_tensor(batch_acts, dtype=torch.float32),
                                   weights=torch.as_tensor(batch_weights, dtype=torch.float32),
                                   old_logps=torch.as_tensor(batch_old_logps, dtype=torch.float32),
                                   )
            loss_pi = loss_pi.item()
            loss_pis.append(loss_pi)
        return np.mean(loss_pis)

    def get_mesh_grid(self):
        X, Y = np.meshgrid(np.arange(-1,1.3,0.1), np.arange(-1,1.3,0.1))
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            print("row: ", i)
            for j in range(X.shape[1]):
                x, y = X[i][j], Y[i][j]
                weight = np.array([x,y])
                loss_pi = self.get_loss_value(weight)
                Z[i][j] = loss_pi

        loss_plot = {'X': X, 'Y': Y, 'Z': Z}
        pickle.dump(loss_plot, open(self.output_dir+"/grid_{}_{}_{}.pkl".format(-1, 1.3, 0.1), "wb"))
        return X, Y, Z


    def plot(self, x, y, z):
        import matplotlib
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'
        matplotlib.rcParams["font.weight"] = "bold"
        matplotlib.rcParams["axes.labelweight"] = "bold"
        fig = plt.figure(figsize=(10, 11))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, y, z,
                        cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=True)
        #wx,wy,wz = get_weights_from_csv(self.output_dir+"/weight_loss.csv")
        #wx,wy,wz = get_weights_from_csv("toy/output/B7R_B7L/weight_loss.csv")
        #ax.scatter3D(wx, wy, wz+120, c="black", s=20)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.zaxis.set_ticklabels([])
        #ax.set_zlabel('z')
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, orientation='horizontal')
        #fig.colorbar(surf, shrink=0.5, aspect=5, orientation='horizontal')
        plt.savefig(self.output_dir+"/plot.pdf", bbox_inches='tight', pad_inches=0)
        plt.show()
        ipdb.set_trace()

    def plot2d(self, x, y, z):
        import matplotlib
        from matplotlib.ticker import MaxNLocator
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'
        matplotlib.rcParams["font.weight"] = "bold"
        matplotlib.rcParams["axes.labelweight"] = "bold"
        fig, ax = plt.subplots(figsize=(6,6))
        cf = ax.contourf(x, y, z,
                               cmap=cm.coolwarm,
                               #linewidth=0,
                               #antialiased=True,
                               #levels=levels,
                          )
        #wx,wy,wz = get_weights_from_csv(self.output_dir+"/weight_loss.csv")
        wx,wy,wz = get_weights_from_csv("toy/output/B7R_B7L/weight_loss.csv")
        indices = np.linspace(0, len(wx)-1, num=13, dtype=int) # for B7L
        #indices = np.linspace(0, len(wx)-1, num=10, dtype=int)

        wx = np.take(wx,indices)
        wy = np.take(wy, indices)

        # for B7L
        indices2 = np.where(wy<=1.0)
        wx = np.take(wx, indices2)
        wy = np.take(wy, indices2)
        ax.set_xlim(-0.98,0.98)
        ax.set_ylim(-0.98,0.98)
        # for B7L

        ax.scatter(wx, wy, c="black",s=100)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        # Add a color bar which maps values to colors.
        #fig.colorbar(cf, ax=ax, shrink=0.5, orientation='horizontal')
        plt.savefig(self.output_dir+"/plot_2d.pdf", bbox_inches='tight', pad_inches=0)
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
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=500)
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
    #vpg.get_loss_value(np.array([1.0,-1.0]))

    # Fine-Tune
    #import os.path as osp
    #fname = osp.join(args.fpath, 'pyt_save', 'model' + str(args.itr) + '_' + str(args.epret) + '.pt')
    #print("args exp name", args.exp_name)
    #print('\n\nLoading from %s\n\n' % fname)
    #vpg.fine_tune(None)

    # Get mesh grid
    #x,y,z = vpg.get_mesh_grid()

    # Plot
    print("plotting")
    with open("toy/output/"+args.exp_name+"/grid_{}_{}_{}.pkl".format(-1,1.3,0.1), "rb") as f:  # for B7L 2d
    #with open("toy/output/"+args.exp_name+"/grid_{}_{}_{}.pkl".format(-1,1,0.1), "rb") as f:
        loss_plot_dict = pickle.load(f)
    x,y,z = loss_plot_dict['X'], loss_plot_dict['Y'], loss_plot_dict['Z']
    vpg.plot2d(x,y,z)






