import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from a2c_ppo_acktr.distributions import Categorical, DiagGaussian, Bernoulli
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def construct_old_policy(obs_shape, action_space, base=None, base_kwargs=None):
    if base_kwargs is None:
        base_kwargs = {}
    if base is None:
        if len(obs_shape) == 3:
            base = CNNBaseOld
        elif len(obs_shape) == 1:
            base = MLPBaseOld
        else:
            raise NotImplementedError

    return base(obs_shape[0], **base_kwargs)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, old_model, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBaseNew
            elif len(obs_shape) == 1:
                base = MLPBaseNew
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], old_model, **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())


            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]


            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBaseOld(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBaseOld, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.main1 = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU())
        self.main2 = nn.Sequential(
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU())
        self.main3 = nn.Sequential(
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten())
        self.main4 = nn.Sequential(
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x1 = self.main1(inputs / 255.0)
        x2 = self.main2(x1)
        x3 = self.main3(x2)
        x4 = self.main4(x3)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs, [x1,x2,x3,x4,x]


class MLPBaseOld(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBaseOld, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))

        self.actor1 = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh())
        self.actor2 = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic1 = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh())
        self.critic2 = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.train()

    def load_model(self, state_dict):
        self.actor1.load_state_dict({'0.weight':state_dict['base.actor.0.weight'],
                                     '0.bias':state_dict['base.actor.0.bias'],})
        self.actor2.load_state_dict({'0.weight':state_dict['base.actor.2.weight'],
                                     '0.bias':state_dict['base.actor.2.bias'],})
        self.critic1.load_state_dict({'0.weight':state_dict['base.critic.0.weight'],
                                     '0.bias':state_dict['base.critic.0.bias'],})
        self.critic2.load_state_dict({'0.weight':state_dict['base.critic.2.weight'],
                                     '0.bias':state_dict['base.critic.2.bias'],})


    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic1 = self.critic1(x)
        hidden_critic = self.critic2(hidden_critic1)
        hidden_actor1 = self.actor1(x)
        hidden_actor = self.actor2(hidden_actor1)

        return hidden_critic1, hidden_critic, hidden_actor1, hidden_actor

class CNNBaseNew(NNBase):
    def __init__(self, num_inputs, old_model, recurrent=False, hidden_size=512):
        super(CNNBaseNew, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))
        self.old_model = old_model
        self.main1 = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU())
        self.main2 = nn.Sequential(
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU())
        self.main3 = nn.Sequential(
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten())
        self.main4 = nn.Sequential(
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x1 = self.main1(inputs / 255.0)
        x2 = self.main2(x1)
        x3 = self.main3(x2)
        x4 = self.main4(x3)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs, [x1,x2,x3,x4,x]


class MLPBaseNew(NNBase):
    def __init__(self, num_inputs, old_model, recurrent=False, hidden_size=64):
        super(MLPBaseNew, self).__init__(recurrent, num_inputs, hidden_size)

        self.old_model = old_model

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))

        self.actor1 = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh())
        self.actor2 = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic1 = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh())
        self.critic2 = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def load_model(self, state_dict):
        import pdb
        pdb.set_trace()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        with torch.no_grad():
            hidden_critic1_old = self.old_model.critic1(x).detach()
            hidden_critic_old = self.old_model.critic2(hidden_critic1_old).detach()
            hidden_actor1_old = self.old_model.actor1(x).detach()
            hidden_actor_old = self.old_model.actor2(hidden_actor1_old).detach()

        hidden_critic1 = self.critic1(x) + hidden_critic1_old
        hidden_critic = self.critic2(hidden_critic1) + hidden_critic_old
        hidden_actor1 = self.actor1(x) + hidden_actor1_old
        hidden_actor = self.actor2(hidden_actor1) + hidden_actor_old

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
