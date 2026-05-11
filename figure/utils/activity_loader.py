
import json
import os

import numpy as np


def _infer_project_root_from_path(path: str):
    abs_path = os.path.abspath(path)
    marker = f"{os.sep}models{os.sep}"
    if marker in abs_path:
        return abs_path.split(marker, 1)[0]
    return None


def _resolve_model_path(model_path: str, metadata_json_path: str):
    if os.path.isabs(model_path):
        return model_path

    metadata_dir = os.path.dirname(os.path.abspath(metadata_json_path))
    candidates = [
        os.path.abspath(os.path.join(metadata_dir, model_path)),
    ]

    project_root = _infer_project_root_from_path(metadata_json_path)
    if project_root is not None:
        candidates.append(os.path.abspath(os.path.join(project_root, model_path)))

    for c in candidates:
        if os.path.exists(c):
            return c

    return candidates[-1]


def _resolve_params_path(model_path: str, metadata_json_path: str):
    params_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), 'parameters.json')
    if os.path.isabs(params_path):
        return params_path

    project_root = _infer_project_root_from_path(metadata_json_path)
    if project_root is not None:
        return os.path.abspath(os.path.join(project_root, params_path))

    return os.path.abspath(params_path)




def load_metadata_and_parameters(metadata_json_path: str):
    with open(metadata_json_path, 'r') as f:
        metadata = json.load(f)
    model_path = _resolve_model_path(metadata['model_path'], metadata_json_path)

    params_path = _resolve_params_path(model_path, metadata_json_path)

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

    base_u_path = os.path.join(os.path.dirname(__file__), 'base_u.npy')
    base_u = np.load(base_u_path, allow_pickle=True)
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

    print(f"data loaded. activity shape: {[a.shape for a in activity]}, efficacy shape: {[e.shape for e in efficacy]}")

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

