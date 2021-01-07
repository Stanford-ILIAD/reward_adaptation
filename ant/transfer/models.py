import torch
import torch.autograd as autograd
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values

class PolicyPNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(PolicyPNN, self).__init__()
        self.affine_list = [nn.Linear(num_inputs, 64),
                            nn.Linear(64, 64)]

        self.action_mean_new = nn.Linear(64, num_outputs)
        self.action_mean_new.weight.data.mul_(0.1)
        self.action_mean_new.bias.data.mul_(0.0)

        self.affine1_new = nn.Linear(num_inputs, 64)
        self.affine2_new = nn.Linear(64, 64)

        self.action_log_std_new = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def load_pretrain(self, state_dict):
        self.affine_list[0].load_state_dict({name.replace('affine1.', ''):state_dict[name] for name in state_dict if 'affine1' in name})
        self.affine_list[1].load_state_dict({name.replace('affine2.', ''):state_dict[name] for name in state_dict if 'affine2' in name})


    def forward(self, x):
        x1 = torch.tanh(self.affine_list[0](x)).detach()
        x2 = torch.tanh(self.affine_list[1](x1)).detach()

        x = torch.tanh(self.affine1_new(x)) + x1
        x = torch.tanh(self.affine2_new(x)) + x2

        action_mean = self.action_mean_new(x)
        action_log_std = self.action_log_std_new.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class ValuePNN(nn.Module):
    def __init__(self, num_inputs):
        super(ValuePNN, self).__init__()
        self.affine_list = [nn.Linear(num_inputs, 64),
                            nn.Linear(64, 64)]
        self.value_head_new = nn.Linear(64, 1)
        self.value_head_new.weight.data.mul_(0.1)
        self.value_head_new.bias.data.mul_(0.0)

        self.affine1_new = nn.Linear(num_inputs, 64)
        self.affine2_new = nn.Linear(64, 64)

    def load_pretrain(self, state_dict):
        self.affine_list[0].load_state_dict({name.replace('affine1.', ''):state_dict[name] for name in state_dict if 'affine1' in name})
        self.affine_list[1].load_state_dict({name.replace('affine2.', ''):state_dict[name] for name in state_dict if 'affine2' in name})
    
    def forward(self, x):
        x1 = torch.tanh(self.affine_list[0](x)).detach()
        x2 = torch.tanh(self.affine_list[1](x1)).detach()

        x = torch.tanh(self.affine1_new(x)) + x1
        x = torch.tanh(self.affine2_new(x)) + x2

        state_values = self.value_head_new(x)
        return state_values

class PolicyL2SP(nn.Module):
    def __init__(self, num_inputs, num_outputs, old_model):
        super(PolicyL2SP, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))
        self.old_state_dict = old_model
        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        return action_mean, action_log_std, action_std


class ValueL2SP(nn.Module):
    def __init__(self, num_inputs, old_model):
        super(ValueL2SP, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)
        self.old_state_dict = old_model

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values
