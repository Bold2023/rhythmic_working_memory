
import json
import os

import numpy as np

from typing import Literal, Tuple




def load_metadata_and_parameters(metadata_json_path: str):
    with open(metadata_json_path, 'r') as f:
        metadata = json.load(f)
    model_path = metadata['model_path']

    params_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), 'parameters.json')

    with open(params_path, 'r') as f:
        params = json.load(f)

    stimulus_num = int(params['stimulus_num'])
    channel_num = int(params['channel_num'])

    time_step = int(params['time_step'])
    fixation_duration = int(params['fixation_duration'])
    stimulus_duration = int(params['stimulus_duration'])
    inter_onset_interval = int(params['inter_onset_interval'])
    sample_duration = stimulus_duration + inter_onset_interval * (stimulus_num - 1)
    delay_duration = int(params['delay_duration'])
    response_duration = int(params['response_duration'])

    hidden_size = int(params['hidden_size'])
    e_size = int(params['e_size'])
    i_size = int(params['i_size'])
    ei_list = np.ones(hidden_size, dtype=np.float32)
    ei_list[-i_size:] = -1

    return metadata, params, (stimulus_num, channel_num, time_step, fixation_duration, stimulus_duration, inter_onset_interval, sample_duration, delay_duration, response_duration, hidden_size, ei_list)


def load_activity_data(npz_paths, activity_type: Literal["neuronal", "synaptic"], details=False):


    rhy_0 = np.load(npz_paths[0], allow_pickle=True)
    rhy_1 = np.load(npz_paths[1], allow_pickle=True)
    arrhy_0 = np.load(npz_paths[2], allow_pickle=True)
    arrhy_1 = np.load(npz_paths[3], allow_pickle=True)

    task = [rhy_0, rhy_1, arrhy_0, arrhy_1]


    if activity_type == "neuronal":
        total_hidden = [x['hidden'] for x in task]

        n_time, n_trial, hidden_size = total_hidden[0].shape

        data = []
        for i in range(4):
            data.append(np.transpose(total_hidden[i], axes=(1, 2, 0)))

    elif activity_type == "synaptic":
        total_syn_x = [x['syn_x'] for x in task]
        total_syn_u = [x['syn_u'] for x in task]

        n_time, n_trial, hidden_size = total_syn_x[0].shape

        base_u = np.load('utils/base_u.npy', allow_pickle=True)
        base_u = base_u[np.newaxis, :, np.newaxis]

        data = []
        syn_x = []
        syn_u = []
        for i in range(4):
            syn_x_transposed = np.transpose(total_syn_x[i], axes=(1, 2, 0))
            syn_u_transposed = np.transpose(total_syn_u[i], axes=(1, 2, 0))
            syn_eff = syn_x_transposed * syn_u_transposed / base_u
            syn_x.append(syn_x_transposed)
            syn_u.append(syn_u_transposed)
            data.append(syn_eff)


    sample_seq = [x['sample_seqs'] for x in task]
    test_seq = [x['test_seqs'] for x in task]
    sample_id = [x['sample_seqs'] for x in task]
    sample_onset_raw = [x['sample_onsets'] for x in task]
    test_onset_raw = [x['test_onsets'] for x in task]

    print(f"successfully loaded data! shape: {len(data)}, {data[0].shape}")

    if details:
        if activity_type == "neuronal":
            return data, sample_id, sample_onset_raw     # list of 4 arrays, each (n_trial, hidden_size, n_time)
        elif activity_type == "synaptic":
            return data, sample_id, sample_onset_raw, (syn_x, syn_u, base_u)
    else:
        return data, sample_id, sample_onset_raw     # list of 4 arrays, each (n_trial, hidden_size, n_time)


def load_activity_data_everything(npz_paths):

    rhy_0 = np.load(npz_paths[0], allow_pickle=True)
    rhy_1 = np.load(npz_paths[1], allow_pickle=True)
    arrhy_0 = np.load(npz_paths[2], allow_pickle=True)
    arrhy_1 = np.load(npz_paths[3], allow_pickle=True)

    task = [rhy_0, rhy_1, arrhy_0, arrhy_1]

    # activity
    total_hidden = [x['hidden'] for x in task]

    activity = []
    for i in range(4):
        activity.append(np.transpose(total_hidden[i], axes=(1, 2, 0)))

    # efficacy
    total_syn_x = [x['syn_x'] for x in task]
    total_syn_u = [x['syn_u'] for x in task]

    base_u = np.load('utils/base_u.npy', allow_pickle=True)
    base_u = base_u[np.newaxis, :, np.newaxis]

    efficacy = []
    syn_x = []
    syn_u = []
    for i in range(4):
        syn_x_transposed = np.transpose(total_syn_x[i], axes=(1, 2, 0))
        syn_u_transposed = np.transpose(total_syn_u[i], axes=(1, 2, 0))
        syn_eff = syn_x_transposed * syn_u_transposed / base_u
        syn_x.append(syn_x_transposed)
        syn_u.append(syn_u_transposed)
        efficacy.append(syn_eff)

    input_activity = [x['input'] for x in task]
    y_activity = [x['y'] for x in task]
    targets = [x['targets'] for x in task]
    sample_seq = [x['sample_seqs'] for x in task]
    test_seq = [x['test_seqs'] for x in task]
    sample_onset_raw = [x['sample_onsets'] for x in task]
    test_onset_raw = [x['test_onsets'] for x in task]

    print(f"data loaded.")

    return activity, efficacy, (syn_x, syn_u, base_u), input_activity, y_activity, targets, sample_seq, test_seq, sample_onset_raw, test_onset_raw
    

def mask_identity_neurons(single_condition_data, std_thres=1e-3):
    # single_condition_data: (n_trial, n_neuron, n_time)

    n_trial, n_neuron, n_time = single_condition_data.shape
    masked_indices = []
    for neuron_idx in range(n_neuron):
        neuron_data = single_condition_data[:, neuron_idx, :]
        std = np.nanstd(neuron_data, axis=0).mean()
        # print(f"{neuron_idx}: {std}")
        if std < std_thres:
            masked_indices.append(neuron_idx)
            single_condition_data[:, neuron_idx, :] = np.nan
    print(f"masked: ({len(masked_indices)}) {masked_indices}")
    return single_condition_data, masked_indices


def normalize_data_zscore(data, do_per_neuron_normalization=True):
    # data: list of 4 arrays, each (n_trial, n_neuron, n_time)

    normed_data = []
    for i in range(len(data)):
        n_trial, n_neuron, n_time = data[i].shape
        mu_baseline = np.mean(data[i], axis=(0, 2), keepdims=True)
        sigma_baseline = np.std(data[i], axis=(0, 2), keepdims=True)
        sigma_baseline[sigma_baseline == 0] = 1e-6 

        data_zscore = (data[i] - mu_baseline) / sigma_baseline

        if do_per_neuron_normalization:
            for neuron_idx in range(n_neuron):
                max_activity = data_zscore[:, neuron_idx, :].max()
                data_zscore[:, neuron_idx, :] /= (max_activity + 1e-6)

        normed_data.append(data_zscore.copy())
        
        
    return normed_data

