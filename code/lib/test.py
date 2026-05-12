import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import random
import pickle

from lib.train import custom_collate_fn, compute_accuracy
from lib.dataset import create_datasets, DMSDataset
from lib.EIRNN import EIRNN
from lib.test_data_generator import create_test_groups
from lib.parameters import DEVICE
from collections import Counter

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_conditions():
    ch0 = dict(keyword_val_pairs=[("change_num", 0)], filter_mode="equal", include_tags=None, exclude_tags=None)
    ch1 = dict(keyword_val_pairs=[("change_num", 1)], filter_mode="equal", include_tags=None, exclude_tags=None)
    ch2 = dict(keyword_val_pairs=[("change_num", 2)], filter_mode="equal", include_tags=None, exclude_tags=None)
    ch3 = dict(keyword_val_pairs=[("change_num", 3)], filter_mode="equal", include_tags=None, exclude_tags=None)
    # return ch0, ch1, ch2
    # return ch0, ch2
    return ch0, ch3

def build_test_configs():
    # ch0, ch1, ch2 = build_conditions()
    # total_filters = [ch0, ch1, ch2]
    # ch0, ch2 = build_conditions()
    # total_filters = [ch0, ch2]
    ch0, ch3 = build_conditions()
    total_filters = [ch0, ch3]
    configs = [
        dict(name='All_Conditions',          filters=total_filters, rhythms=[("rhythmic","semirhythmic"), ("arrhythmic","semirhythmic")]),
        dict(name='Rhythmic_Only',           filters=total_filters, rhythms=[("rhythmic","semirhythmic")]),
        dict(name='Arrhythmic_Only',         filters=total_filters, rhythms=[("arrhythmic","semirhythmic")]),

        dict(name='All_Rhythms_Change_0',    filters=[ch0], rhythms=[("rhythmic","semirhythmic"), ("arrhythmic","semirhythmic")]),
        dict(name='Rhythmic_Change_0',       filters=[ch0], rhythms=[("rhythmic","semirhythmic")]),
        dict(name='Arrhythmic_Change_0',     filters=[ch0], rhythms=[("arrhythmic","semirhythmic")]),

        # dict(name='All_Rhythms_Change_1',    filters=[ch1], rhythms=[("rhythmic","semirhythmic"), ("arrhythmic","semirhythmic")]),
        # dict(name='Rhythmic_Change_1',       filters=[ch1], rhythms=[("rhythmic","semirhythmic")]),
        # dict(name='Arrhythmic_Change_1',     filters=[ch1], rhythms=[("arrhythmic","semirhythmic")]),

        # dict(name='All_Rhythms_Change_2',    filters=[ch2], rhythms=[("rhythmic","semirhythmic"), ("arrhythmic","semirhythmic")]),
        # dict(name='Rhythmic_Change_2',       filters=[ch2], rhythms=[("rhythmic","semirhythmic")]),
        # dict(name='Arrhythmic_Change_2',     filters=[ch2], rhythms=[("arrhythmic","semirhythmic")]),

        dict(name='All_Rhythms_Change_3',    filters=[ch3], rhythms=[("rhythmic","semirhythmic"), ("arrhythmic","semirhythmic")]),
        dict(name='Rhythmic_Change_3',       filters=[ch3], rhythms=[("rhythmic","semirhythmic")]),
        dict(name='Arrhythmic_Change_3',     filters=[ch3], rhythms=[("arrhythmic","semirhythmic")]),
    ]
    return configs


def generate_dataset(datasets, test_size, filters, rhythms, 
                    share_arrhythmic_sample_onsets=False,
                    share_ch3_test_seqs=False):
    num_groups = test_size // 6
    groups = create_test_groups(
        num_groups=num_groups, 
        datasets=datasets, 
        device='cpu',
        share_arrhythmic_sample_onsets=share_arrhythmic_sample_onsets,
        share_ch3_test_seqs=share_ch3_test_seqs
    )
    
    large_dataset = []
    for group in groups:
        for trial in group['trials'].values():
            large_dataset.append(trial)
    
    random.shuffle(large_dataset)

    print(f'Balanced dataset size: {len(large_dataset)}')
    balanced_conditions = [(d['sample_rhythmicity'], d['change_num']) for d in large_dataset]
    print('Balanced condition dist:', dict(Counter(balanced_conditions)))

    eval_dataset = DMSDataset(large_dataset, device=DEVICE)

    return eval_dataset

def evaluate_and_get_details(model, batch, saved_params, save_activations=False, save_dir=None, shuffle_time_step=None):

    inputs = batch['input'].to(DEVICE)
    targets = batch['label'].to(DEVICE)
    metadata_list = batch['metadata']

    B, T = targets.shape[0], targets.shape[1]

    res_loss = saved_params.get('res_loss_mask', 5)
    loss_mask = torch.full((B, T), saved_params.get('non_res_loss_mask', 1), device=DEVICE)

    for b in range(B):
        response_index = np.array(metadata_list[b]['response_index'], dtype=int)
        grace_mask = response_index[:saved_params.get('grace_steps', 1)]
        # grace_mask = response_index[:2]
        loss_mask[b, response_index] = res_loss
        loss_mask[b, grace_mask] = 0
    
    # Prepare shuffle_indices if needed
    shuffle_indices = None
    if shuffle_time_step is not None:
         shuffle_indices = torch.full((B,), shuffle_time_step, device=DEVICE, dtype=torch.long)

    inputs = inputs.transpose(0, 1)
    targets = targets.transpose(0, 1)
    loss_mask = loss_mask.transpose(0, 1)

    if saved_params.get('use_stsp', True):
        hidden, syn_x, syn_u, y = model(inputs, shuffle_indices=shuffle_indices)
    else:
        hidden, y = model(inputs)

    y_pred = torch.argmax(y, dim=-1)
    target_class = torch.argmax(targets, dim=-1)
    correct = (y_pred == target_class)
    response_mask = (loss_mask == res_loss)

    detailed_results = []

    for b in range(B):
        resp_idx = response_mask[:, b].nonzero(as_tuple=True)[0]
        is_correct = False
        if len(resp_idx) > 0 and correct[resp_idx, b].all():
            is_correct = True

        sample_activation_data = {
            'input': inputs[:, b:b+1, :].detach().cpu().numpy(),           # (T, 1, input_size)
            'hidden': hidden[:, b:b+1, :].detach().cpu().numpy(),          # (T, 1, hidden_size)
            'y': y[:, b:b+1, :].detach().cpu().numpy(),                    # (T, 1, output_size)
            'targets': targets[:, b:b+1, :].detach().cpu().numpy(),        # (T, 1, output_size)
        }
        
        if saved_params.get('use_stsp', True):
            sample_activation_data['syn_x'] = syn_x[:, b:b+1, :].detach().cpu().numpy()
            sample_activation_data['syn_u'] = syn_u[:, b:b+1, :].detach().cpu().numpy()
        
        sample_activation_data['metadata'] = {
            'time_steps': T,
            'input_size': inputs.shape[-1],
            'hidden_size': hidden.shape[-1],
            'output_size': y.shape[-1],
            'sample_rhythmicity': metadata_list[b]['sample_rhythmicity'],
            'test_rhythmicity': metadata_list[b]['test_rhythmicity'],
            'change_num': metadata_list[b]['change_num'],
        }
        
        detailed_results.append({
            'is_correct': is_correct,
            'metadata': metadata_list[b],
            'activation_data': sample_activation_data
        })
    
    return detailed_results


def test_model(model_path, test_size=4000, save_activations=False,
            share_arrhythmic_sample_onsets=False,
              share_ch3_test_seqs=False,
              use_stsp=None,
              shuffle_time_step=None):

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    
    save_dir = os.path.dirname(os.path.dirname(model_path))
    instances_path = os.path.join(save_dir, 'instances.pkl')
    if not os.path.exists(instances_path):
        raise FileNotFoundError(f"instances.pkl not found at {instances_path}")
    with open(instances_path, 'rb') as f:
        instances = pickle.load(f)
    datasets = instances['datasets']

    checkpoint = torch.load(model_path, map_location=DEVICE)
    saved_params = checkpoint['params']

    if use_stsp is not None:
        saved_params['use_stsp'] = use_stsp
        print(f"use_stsp: {use_stsp}")

    print(f'Loaded epoch: {checkpoint.get("epoch")}')
    model_params = saved_params.copy()

    print(save_dir)
    print(("test_variations" in saved_params.keys()))
    print((datasets.dataset_generator.onset_generator.test_variations))

    # print(1/0)

    model_params['is_noise'] = True
    model = EIRNN(**model_params).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    large_dataset = generate_dataset(
        datasets,
        test_size,
        filters = None,
        rhythms = [("rhythmic","semirhythmic"), ("arrhythmic","semirhythmic")],
        share_arrhythmic_sample_onsets=share_arrhythmic_sample_onsets,
        share_ch3_test_seqs=share_ch3_test_seqs
    )

    print(f'Total dataset size for evaluation: {len(large_dataset)}')

    print((datasets.dataset_generator.onset_generator.test_variations))

    total_loader = DataLoader(
        large_dataset,
        batch_size=len(large_dataset),
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    activation_save_dir = None
    if save_activations:
        activation_save_dir = os.path.join(save_dir, 'activations')
        os.makedirs(activation_save_dir, exist_ok=True)
        print(f"Activations will be saved to: {activation_save_dir}\n")

    
    all_sample_results = []
    with torch.no_grad():
        for batch in total_loader:
            print((datasets.dataset_generator.onset_generator.test_variations))
            all_sample_results = evaluate_and_get_details(model, batch, saved_params,
                save_activations=save_activations,
                save_dir=activation_save_dir,
                shuffle_time_step=shuffle_time_step)

    configs = build_test_configs()
    results = {}

    save_config_names = [
        'Rhythmic_Change_0', 'Arrhythmic_Change_0',
        'Rhythmic_Change_2', 'Arrhythmic_Change_2',
    ]
    all_config_names = [cfg['name'] for cfg in configs]
    if 'Rhythmic_Change_1' in all_config_names:
        save_config_names.extend(['Rhythmic_change_1', 'Arrhythmic_change_1'])


    for cfg in configs:
        print(f'\n=== {cfg["name"]} ===')

        filtered_results = [
            res for res in all_sample_results
            if (
                any(
                    all(res['metadata'][kvp[0]] == kvp[1] for kvp in f['keyword_val_pairs'])
                    for f in cfg['filters']
                ) if cfg['filters'] else True
            ) and (
                (res['metadata'].get('sample_rhythmicity'), res['metadata'].get('test_rhythmicity')) in cfg['rhythms']
                if cfg['rhythms'] else True
            )
        ]
        
        dataset_size = len(filtered_results)
        print(f'Dataset size: {dataset_size}')
        if dataset_size == 0:
            results[cfg['name']] = 0.0
            print(f'[{cfg["name"]}] Response Accuracy: 0.0000')
            continue

        change_nums = [d['metadata']['change_num'] for d in filtered_results]
        print('change_num dist:', dict(Counter(change_nums)))
        rhythms = [(d['metadata'].get('sample_rhythmicity'),
                    d['metadata'].get('test_rhythmicity')) for d in filtered_results]
        print('rhythmicity dist:', dict(Counter(rhythms)))


        # config 保存 npz 文件
        if save_activations and activation_save_dir is not None and filtered_results and cfg['name'] in save_config_names:

            config_activation_data = {
                'input': np.concatenate([res['activation_data']['input'] for res in filtered_results], axis=1),
                'hidden': np.concatenate([res['activation_data']['hidden'] for res in filtered_results], axis=1),
                'y': np.concatenate([res['activation_data']['y'] for res in filtered_results], axis=1),
                'targets': np.concatenate([res['activation_data']['targets'] for res in filtered_results], axis=1),
            }
            
            if saved_params.get('use_stsp', True):
                config_activation_data['syn_x'] = np.concatenate([res['activation_data']['syn_x'] for res in filtered_results], axis=1)
                config_activation_data['syn_u'] = np.concatenate([res['activation_data']['syn_u'] for res in filtered_results], axis=1)
            
            config_activation_data['metadata'] = {
                'num_samples': dataset_size,
                'time_steps': filtered_results[0]['activation_data']['metadata']['time_steps'],
                'input_size': filtered_results[0]['activation_data']['metadata']['input_size'],
                'hidden_size': filtered_results[0]['activation_data']['metadata']['hidden_size'],
                'output_size': filtered_results[0]['activation_data']['metadata']['output_size'],
                'sample_rhythmicity': [res['metadata']['sample_rhythmicity'] for res in filtered_results],
                'test_rhythmicity': [res['metadata']['test_rhythmicity'] for res in filtered_results],
                'change_num': [res['metadata']['change_num'] for res in filtered_results],
                'is_correct': [res['is_correct'] for res in filtered_results],
            }
            
            filename = f"activity_{cfg['name']}.npz"
            filepath = os.path.join(activation_save_dir, filename)
            np.savez(filepath, **config_activation_data)
            print(f"Saved activations: {filename}")
            print(f"input shape: {config_activation_data['input'].shape}")
            print(f"hidden shape: {config_activation_data['hidden'].shape}")
            print(f"output shape: {config_activation_data['y'].shape}")
            if saved_params.get('use_stsp', True):
                print(f"syn_x shape: {config_activation_data['syn_x'].shape}")
                print(f"syn_u shape: {config_activation_data['syn_u'].shape}")

        correct_count = sum(1 for res in filtered_results if res['is_correct'])
        accuracy = correct_count / dataset_size if dataset_size > 0 else 0.0
        
        results[cfg['name']] = accuracy
        print(f'[{cfg["name"]}] Response Accuracy: {accuracy:.4f}')

    print('\n---------------------------------------------------')
    print('Comprehensive Test Results Summary')
    print('---------------------------------------------------')
    for k, v in results.items():
        print(f'{k:<28s} | Response Accuracy: {v:.4f}')
    print('---------------------------------------------------')
    return results
