
import torch

# device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training parameters
TRAINING_PARAMS = {
    'batch_size': 2048, 
    'num_epochs': 3000, 
    'save_interval': 200, 
    'learning_rate': 0.02, 
    'clip_grad': 1.0, 
    'save_parent_dir': 'models/', 
    'perf_penalty' : 1.0, 
    'spike_penalty': 1e-2, 
    'weight_penalty': 1e-3, 
    'grace_steps' : 1, 
    'res_loss_mask': 5, 
    'non_res_loss_mask': 1, 
}

# dataset parameters
DATASET_PARAMS = {
    'stimulus_num': 6,
    'train_size': 8192,
    'test_size': 0,
    'change_nums': [0, 3],
    'allow_repeat': False,
    'permitted_movements': None,
    'permitted_movement_directions': ["positive"],

    'arrhythmic_ratio': 0.6,
    'semirhythmic_ratio': 0.3,
    'rhythmicity_pair_conditions': [
        ("rhythmic", "semirhythmic"), 
        ("arrhythmic", "semirhythmic")
    ],

    'time_step': 10,
    'fixation_duration': 500,
    'delay_duration': 1000,
    'response_duration': 100,

    'stimulus_duration': 100,
    'inter_onset_interval': 350,

    'channel_num': 6,
    'direction_num': 6,
    'channel_to_direction_mapping': None,
    'tuning_neuron_num_per_dir': 5,
    'tuning_height': 4,
    'tuning_kappa': 2,

    # 'test_variations': [-2, 3, 1, -4],

    'use_all_pairs': False,
}

# filter conditions
change_0_condition = dict(
    keyword_val_pairs=[("change_num", 0)],
)
# change_1_condition = dict(
#     keyword_val_pairs=[("change_num", 1)],
# )
# change_2_condition = dict(
#     keyword_val_pairs=[("change_num", 2)],
# )
change_3_condition = dict(
    keyword_val_pairs=[("change_num", 3)],
)
DATASET_PARAMS['filter_conditions'] = ([change_0_condition] + [change_3_condition])

# model parameters
MODEL_PARAMS = {
    'hidden_size': 100,
    'output_size': 3,
    'e_prop': 0.8,
    'connection_prob': 1.0,
    'membrane_time_constant': 100,
    'noise_rnn': 0.25,
    'noise_in': 0.05,
    'use_stsp': True,
    'synapse_config': 'full',
    'is_noise': True,
}

