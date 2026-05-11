
import torch
import numpy as np


def initialize_PARAMS(PARAMS, DEVICE):

    PARAMS['tuning_neuron_num'] = PARAMS['direction_num'] * PARAMS['tuning_neuron_num_per_dir']

    PARAMS['input_size'] = PARAMS['tuning_neuron_num'] + 1
    PARAMS['dt'] = PARAMS['time_step']

    # EI size & membrane constant time
    PARAMS['alpha_neuron'] = PARAMS['dt'] / PARAMS['membrane_time_constant']
    PARAMS['dt_sec'] = PARAMS['dt'] / 1000.0
    PARAMS['e_size'] = int(PARAMS['hidden_size'] * PARAMS['e_prop'])
    PARAMS['i_size'] = PARAMS['hidden_size'] - PARAMS['e_size']

    # EI mask
    PARAMS['EI_list'] = np.ones(PARAMS['hidden_size'], dtype=np.float32)
    PARAMS['EI_list'][-PARAMS['i_size']:] = -1
    PARAMS['ind_inh'] = np.where(PARAMS['EI_list'] == -1)[0]
    PARAMS['EI_MATRIX'] = np.expand_dims(PARAMS['EI_list'], 0).repeat(PARAMS['hidden_size'], axis=0).T

    # connection mask
    PARAMS['w_input_mask'] = np.ones((PARAMS['input_size'], PARAMS['hidden_size']), dtype=np.float32)
    PARAMS['w_input_mask'][:, PARAMS['ind_inh']] = 0

    # RNN mask
    PARAMS['w_rnn_mask'] = np.ones((PARAMS['hidden_size'], PARAMS['hidden_size']), dtype=np.float32) - np.eye(PARAMS['hidden_size'])
    PARAMS['w_output_mask'] = np.ones((PARAMS['hidden_size'], PARAMS['output_size']), dtype=np.float32)
    PARAMS['w_output_mask'][PARAMS['ind_inh'], :] = 0

    PARAMS['EI_MATRIX'] = torch.tensor(PARAMS['EI_MATRIX'], dtype=torch.float32).to(DEVICE)
    PARAMS['w_input_mask'] = torch.tensor(PARAMS['w_input_mask'], dtype=torch.float32).to(DEVICE)
    PARAMS['w_rnn_mask'] = torch.tensor(PARAMS['w_rnn_mask'], dtype=torch.float32).to(DEVICE)
    PARAMS['w_output_mask'] = torch.tensor(PARAMS['w_output_mask'], dtype=torch.float32).to(DEVICE)


    def initialize(dims, connection_prob, shape=0.1, scale=1.0):
        w = np.random.gamma(shape, scale, size=dims)
        w *= (np.random.rand(*dims) < connection_prob)
        return np.float32(w)

    init_input_weight = initialize([PARAMS['input_size'], PARAMS['hidden_size']], PARAMS['connection_prob'], shape=0.2, scale=1.0)
    init_rnn_weight = initialize([PARAMS['hidden_size'], PARAMS['hidden_size']], PARAMS['connection_prob'])
    init_output_weight = initialize([PARAMS['hidden_size'], PARAMS['output_size']], PARAMS['connection_prob'])

    PARAMS['init_input_weight'] = torch.tensor(init_input_weight, dtype=torch.float32, device=DEVICE) * PARAMS['w_input_mask']
    PARAMS['init_rnn_weight'] = torch.tensor(init_rnn_weight, dtype=torch.float32, device=DEVICE) * PARAMS['EI_MATRIX'] * PARAMS['w_rnn_mask']
    PARAMS['init_output_weight'] = torch.tensor(init_output_weight, dtype=torch.float32, device=DEVICE) * PARAMS['w_output_mask']
    PARAMS['init_input_b'] = torch.zeros(PARAMS['hidden_size'], device=DEVICE)
    PARAMS['init_rnn_b'] = torch.zeros(PARAMS['hidden_size'], device=DEVICE)
    PARAMS['init_output_b'] = torch.zeros(PARAMS['output_size'], device=DEVICE)

    # STSP init
    PARAMS['init_syn_x'] = torch.ones(1, PARAMS['hidden_size']).to(DEVICE)
    PARAMS['init_syn_u'] = 0.3 * torch.ones(1, PARAMS['hidden_size']).to(DEVICE)
    PARAMS['alpha_stf'] = torch.ones(1, PARAMS['hidden_size']).to(DEVICE)
    PARAMS['alpha_std'] = torch.ones(1, PARAMS['hidden_size']).to(DEVICE)
    PARAMS['U'] = torch.ones(1, PARAMS['hidden_size']).to(DEVICE)

    synaptic_configurations = ['facilitating' if i%2==0 else 'depressing' for i in range(PARAMS['hidden_size'])]

    for i in range(PARAMS['hidden_size']):
        if synaptic_configurations[i] == 'facilitating':
            PARAMS['alpha_stf'][0, i] = PARAMS['dt'] / 1500
            PARAMS['alpha_std'][0, i] = PARAMS['dt'] / 200
            PARAMS['U'][0, i] = 0.15
            PARAMS['init_syn_u'][:, i] = PARAMS['U'][0, i]
        elif synaptic_configurations[i] == 'depressing':
            PARAMS['alpha_stf'][0, i] = PARAMS['dt'] / 200
            PARAMS['alpha_std'][0, i] = PARAMS['dt'] / 1500
            PARAMS['U'][0, i] = 0.45
            PARAMS['init_syn_u'][:, i] = PARAMS['U'][0, i]
    
    return PARAMS
