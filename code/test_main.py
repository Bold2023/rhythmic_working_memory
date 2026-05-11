import os
import pickle
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from collections import defaultdict

from lib.train import custom_collate_fn
from lib.dataset import create_datasets, DMSDataset
from lib.EIRNN import EIRNN
from lib.parameters import DEVICE


model_path = r'YOUR_MODEL_PATH_HERE'  # Replace with your model path


def save_all_pairs_activity(model_path, output_dir=None, batch_size=100):

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    
    save_dir = os.path.dirname(os.path.dirname(model_path))
    instances_path = os.path.join(save_dir, 'instances.pkl')
    if not os.path.exists(instances_path):
        raise FileNotFoundError(f"instances.pkl not found at {instances_path}")
    
    with open(instances_path, 'rb') as f:
        instances = pickle.load(f)
    datasets = instances['datasets']

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    saved_params = checkpoint['params']
    print(f'Loaded epoch: {checkpoint.get("epoch")}')

    model_params = saved_params.copy()
    model_params['is_noise'] = False
    model = EIRNN(**model_params).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ch0 = dict(keyword_val_pairs=[("change_num", 0)], filter_mode="equal", 
               include_tags=None, exclude_tags=None)
    ch3 = dict(keyword_val_pairs=[("change_num", 3)], filter_mode="equal",
               include_tags=None, exclude_tags=None)
    
    filter_conditions = [ch0, ch3]
    rhythmicity_pair_conditions = [("rhythmic", "semirhythmic"), ("arrhythmic", "semirhythmic")]

    train_dataset, test_dataset = datasets.create(
        train_size=1,
        test_size=0,
        filter_conditions=filter_conditions,
        rhythmicity_pair_conditions=rhythmicity_pair_conditions,
        device='cpu',
        use_all_pairs=True
    )

    all_trial_dicts = []
    for condition_idx, condition_dicts in datasets.cached_all_pairs_train_dicts.items():
        all_trial_dicts.extend(condition_dicts)

    print(f"Total trials to evaluate: {len(all_trial_dicts)}")

    full_dataset = DMSDataset(all_trial_dicts, device=DEVICE)
    dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    grouped_results = defaultdict(lambda: {
        'inputs': [],
        'hiddens': [],
        'ys': [],
        'targets': [],
        'syn_xs': [],
        'syn_us': [],
        'sample_seqs': [],
        'test_seqs': [],
        'sample_onsets': [],
        'test_onsets': [],
        'is_correct': []
    })

    print('Start testing all pairs...')
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if (batch_idx + 1) % 10 == 0:
                print(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")
            
            inputs = batch['input'].to(DEVICE)
            targets = batch['label'].to(DEVICE)
            metadata_list = batch['metadata']

            B, T = targets.shape[0], targets.shape[1]

            res_loss = saved_params.get('res_loss_mask', 5)
            loss_mask = torch.full((B, T), saved_params.get('non_res_loss_mask', 1), device=DEVICE)
            
            for b in range(B):
                response_index = np.array(metadata_list[b]['response_index'], dtype=int)
                grace_mask = response_index[:saved_params.get('grace_steps', 1)]
                loss_mask[b, response_index] = res_loss
                loss_mask[b, grace_mask] = 0

            inputs = inputs.transpose(0, 1)
            targets = targets.transpose(0, 1)
            loss_mask = loss_mask.transpose(0, 1)

            if saved_params.get('use_stsp', True):
                hidden, syn_x, syn_u, y = model(inputs)
            else:
                hidden, y = model(inputs)
                syn_x, syn_u = None, None

            y_pred = torch.argmax(y, dim=-1)
            target_class = torch.argmax(targets, dim=-1)
            correct = (y_pred == target_class)
            response_mask = (loss_mask == res_loss)

            # print(f"Metadata keys: {metadata_list[0].keys()}")

            for b in range(B):
                resp_idx = response_mask[:, b].nonzero(as_tuple=True)[0]
                is_correct = False
                if len(resp_idx) > 0 and correct[resp_idx, b].all():
                    is_correct = True

                sample_rhy = metadata_list[b]['sample_rhythmicity']
                test_rhy = metadata_list[b]['test_rhythmicity']
                change_num = metadata_list[b]['change_num']

                if sample_rhy == 'rhythmic' and test_rhy == 'semirhythmic':
                    if change_num == 0:
                        cond_key = 'rhy_ch0'
                    elif change_num == 3:
                        cond_key = 'rhy_ch3'
                    else:
                        continue
                elif sample_rhy == 'arrhythmic' and test_rhy == 'semirhythmic':
                    if change_num == 0:
                        cond_key = 'arrhy_ch0'
                    elif change_num == 3:
                        cond_key = 'arrhy_ch3'
                    else:
                        continue
                else:
                    continue

                grouped_results[cond_key]['inputs'].append(inputs[:, b:b+1, :].detach().cpu().numpy())
                grouped_results[cond_key]['hiddens'].append(hidden[:, b:b+1, :].detach().cpu().numpy())
                grouped_results[cond_key]['ys'].append(y[:, b:b+1, :].detach().cpu().numpy())
                grouped_results[cond_key]['targets'].append(targets[:, b:b+1, :].detach().cpu().numpy())
                
                if saved_params.get('use_stsp', True):
                    grouped_results[cond_key]['syn_xs'].append(syn_x[:, b:b+1, :].detach().cpu().numpy())
                    grouped_results[cond_key]['syn_us'].append(syn_u[:, b:b+1, :].detach().cpu().numpy())
                
                seq_pair = metadata_list[b]['seq_pair']
                grouped_results[cond_key]['sample_seqs'].append(seq_pair[0])
                grouped_results[cond_key]['test_seqs'].append(seq_pair[1])

                onset_pair = metadata_list[b]['onset_pair']
                grouped_results[cond_key]['sample_onsets'].append(onset_pair[0])
                grouped_results[cond_key]['test_onsets'].append(onset_pair[1])

                grouped_results[cond_key]['is_correct'].append(is_correct)
    

    if output_dir is None:
        output_dir = os.path.join(save_dir, 'activity_all_pairs')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSaving activations...")
    for cond_key, cond_data in grouped_results.items():
        if len(cond_data['inputs']) == 0:
            print(f"Warning: No data for condition {cond_key}")
            continue

        save_data = {
            'input': np.concatenate(cond_data['inputs'], axis=1),
            'hidden': np.concatenate(cond_data['hiddens'], axis=1),
            'y': np.concatenate(cond_data['ys'], axis=1),
            'targets': np.concatenate(cond_data['targets'], axis=1),
            'sample_seqs': cond_data['sample_seqs'],
            'test_seqs': cond_data['test_seqs'],
            'sample_onsets': cond_data['sample_onsets'],
            'test_onsets': cond_data['test_onsets'],
            'is_correct': cond_data['is_correct'],
        }

        if saved_params.get('use_stsp', True):
            save_data['syn_x'] = np.concatenate(cond_data['syn_xs'], axis=1)
            save_data['syn_u'] = np.concatenate(cond_data['syn_us'], axis=1)

        filename = f'activity_{cond_key}_all_pairs.npz'
        filepath = os.path.join(output_dir, filename)
        np.savez(filepath, **save_data)

        n_samples = len(cond_data['is_correct'])
        n_correct = sum(cond_data['is_correct'])
        accuracy = n_correct / n_samples * 100 if n_samples > 0 else 0

        print(f'\nSaved {filename}')
        print(f'  Num samples: {n_samples}')
        print(f'  Correct: {n_correct}/{n_samples} ({accuracy:.2f}%)')
        print(f'  Input shape: {save_data["input"].shape}')
        print(f'  Hidden shape: {save_data["hidden"].shape}')
        if saved_params.get('use_stsp', True):
            print(f'  syn_x shape: {save_data["syn_x"].shape}')
            print(f'  syn_u shape: {save_data["syn_u"].shape}')

        del save_data
    
    metadata = {
        'model_path': model_path,
        'total_trials': len(all_trial_dicts),
        'conditions': list(grouped_results.keys()),
    }
    
    for cond_key in grouped_results.keys():
        metadata[f'{cond_key}_num_samples'] = len(grouped_results[cond_key]['is_correct'])
        metadata[f'{cond_key}_accuracy'] = (
            sum(grouped_results[cond_key]['is_correct']) / 
            len(grouped_results[cond_key]['is_correct']) * 100
        )

    metadata_path = os.path.join(output_dir, 'metadata_all_pairs.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f'\nSaved metadata to {metadata_path}')
    print(f'Output directory: {output_dir}')
    
    return output_dir

if __name__ == '__main__':
    output_dir = save_all_pairs_activity(model_path, batch_size=100)


