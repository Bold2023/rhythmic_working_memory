import math
import torch
import numpy as np
import torch.nn as nn

from torch.nn import init
from torch.nn import functional as F


class InputLayer(nn.Module):
    def __init__(self, **params):
        super().__init__()

        if 'init_input_b' in params.keys():
            self.bias = nn.Parameter(params['init_input_b'])
        else:
            self.bias = nn.Parameter(torch.zeros(params['hidden_size']))
            
        if 'init_input_weight' in params.keys():
            self.weight = nn.Parameter(params['init_input_weight']) 
        else:
            self.weight = nn.Parameter(torch.zeros(params['input_size'], params['hidden_size']))
            self.reset_parameters() 
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #self.weight = torch.abs(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        return F.linear(input, self.weight.t().type(torch.float32), self.bias)
    
    
class RnnLayer(nn.Module):
    def __init__(self, **params):
        super().__init__()

        self.e_size = params['e_size']
        self.i_size = params['i_size']
        
        if 'init_rnn_b' in params.keys():
            self.bias = nn.Parameter(params['init_rnn_b'])
        else:
            self.bias = nn.Parameter(torch.zeros(params['hidden_size']))
        
        if 'init_rnn_weight' in params.keys():
            self.weight = nn.Parameter(params['init_rnn_weight'])
        else:
            self.weight = nn.Parameter(torch.zeros(params['hidden_size'], params['hidden_size']))
            self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data[:, :self.e_size] /= (self.e_size / self.i_size)
            #self.weight = torch.abs(self.weight) * self.mask
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        return F.linear(input, self.weight.t().type(torch.float32), self.bias)

        
class OutputLayer(nn.Module):
    def __init__(self, **params):
        super().__init__()
        
        if 'init_output_b' in params.keys():
            self.bias = nn.Parameter(params['init_output_b'])
        else:
            self.bias = nn.Parameter(torch.zeros(params['output_size']))
        
        if 'init_output_weight' in params.keys():
            self.weight = nn.Parameter(params['init_output_weight'])
        else:
            self.weight = nn.Parameter(torch.zeros(params['e_size'], params['output_size']))
            self.reset_parameters()
            
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        Output = F.linear(input, self.weight.t().type(torch.float32), self.bias)

        return Output


class EIRNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.params = kwargs

        #self.par = par
        self.input2h = InputLayer(**self.params)
        self.h2h = RnnLayer(**self.params)
        self.h2output = OutputLayer(**self.params)
        self.init_hidden = nn.Parameter(torch.empty(1, self.params['hidden_size']).uniform_(-0.2, 0.2).abs())
        self.is_noise = self.params['is_noise']

        if self.params['synapse_config'] == 'full':
            self.one_minus_alphaneuron = 1 - self.params['alpha_neuron']
        else:
            self.one_minus_alphaneuron = [1 - x for x in self.params['alpha_neuron']]

        self.device = self.params['device']

    def init_syn_x_state(self, batch_size):
        init_syn_x = self.params['init_syn_x'].expand(batch_size, self.params['hidden_size']).contiguous().to(self.device)
        return init_syn_x
    
    def init_syn_u_state(self, batch_size):
        init_syn_u = self.params['init_syn_u'].expand(batch_size, self.params['hidden_size']).contiguous().to(self.device)
        return init_syn_u

    def recurrence_syn(self, input, hidden, syn_x, syn_u):

        syn_x = syn_x + (self.params['alpha_std'] * (1 - syn_x) - self.params['dt_sec'] * syn_u * syn_x * hidden)
        syn_u = syn_u + (self.params['alpha_stf'] * (self.params['U'] - syn_u) + self.params['dt_sec'] * self.params['U'] * (1 - syn_u) * hidden)

        # print(f'syn_x.shape: {syn_x.shape}, syn_u.shape: {syn_u.shape}, hidden.shape: {hidden.shape}')

        syn_x = torch.minimum(torch.tensor([1]).to(self.device), torch.relu(syn_x))
        syn_u = torch.minimum(torch.tensor([1]).to(self.device), torch.relu(syn_u))

        h_post = syn_u * syn_x * hidden

        hidden = torch.relu(hidden * self.one_minus_alphaneuron
                + torch.relu(self.input2h(input + self.noise_in)
                + self.h2h(h_post) + self.noise_rnn) * self.params['alpha_neuron'])

        return hidden, syn_x, syn_u

    def recurrence(self, input, hidden):

        hidden = torch.relu(torch.sigmoid(
            hidden * self.one_minus_alphaneuron
            + torch.relu(self.input2h(input + self.noise_in)
            + self.h2h(hidden) + self.noise_rnn) * self.params['alpha_neuron']
        ))

        return hidden

    def _forward_syn(self, input, hidden=None, syn_x=None, syn_u=None):
        batch_size = input.shape[1]
 
        if hidden is None:
            #self.init_h.expand(*state_size).contiguous()
            hidden = self.init_hidden.expand(batch_size, self.params['hidden_size']).contiguous().to(self.device)
        if syn_x is None:
            syn_x = self.init_syn_x_state(batch_size)
        if syn_u is None:
            syn_u = self.init_syn_u_state(batch_size)

        self.hidden = []
        self.y = []
        self.syn_x = []
        self.syn_u = []

        for step in input:
            if self.is_noise:
                self.noise_in = torch.randn(step.shape).to(self.device) * self.params['noise_in']
                self.noise_rnn = torch.randn((batch_size, self.params['hidden_size'])).to(self.device) * self.params['noise_rnn']
            else:
                self.noise_in = torch.zeros(step.shape, dtype=torch.float32).to(self.device)
                self.noise_rnn = torch.zeros((batch_size, self.params['hidden_size']), dtype=torch.float32).to(self.device)

            hidden, syn_x, syn_u = self.recurrence_syn(step, hidden, syn_x, syn_u)
     
            self.hidden.append(hidden)
            self.syn_x.append(syn_x)
            self.syn_u.append(syn_u)
            self.y.append(self.h2output(hidden))
        
        self.hidden = torch.stack(self.hidden)
        self.syn_x = torch.stack(self.syn_x)
        self.syn_u = torch.stack(self.syn_u)
        self.y = torch.stack(self.y)

        return self.hidden, self.syn_x, self.syn_u, self.y
    
    def _forward(self, input, hidden=None):
        batch_size = input.shape[1]
 
        if hidden is None:
            #self.init_h.expand(*state_size).contiguous()
            hidden = self.init_hidden.expand(batch_size, self.params['hidden_size']).contiguous().to(self.device)

        self.hidden = []
        self.y = []

        for step in input:
            if self.is_noise:
                self.noise_in = torch.randn(step.shape).to(self.device) * self.params['noise_in']
                self.noise_rnn = torch.randn((batch_size, self.params['hidden_size'])).to(self.device) * self.params['noise_rnn']
            else:
                self.noise_in = torch.zeros(step.shape, dtype=torch.float32).to(self.device)
                self.noise_rnn = torch.zeros((batch_size, self.params['hidden_size']), dtype=torch.float32).to(self.device)

            hidden = self.recurrence(step, hidden)
     
            self.hidden.append(hidden)
            self.y.append(self.h2output(hidden))
        
        self.hidden = torch.stack(self.hidden)
        self.y = torch.stack(self.y)

        return self.hidden, self.y
     
    def forward(self, input, hidden=None, syn_x=None, syn_u=None):
        if self.params['use_stsp']:
            return self._forward_syn(input, hidden, syn_x, syn_u)
        else:
            return self._forward(input, hidden)


