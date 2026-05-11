import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import pickle
import json

from lib.dataset import create_datasets
# import matplotlib.pyplot as plt


print("train version: final")
print(" ")


def custom_collate_fn(batch):
    inputs = torch.stack([item['input'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    metadata_list = [item['metadata'] for item in batch]

    return {
        'input': inputs,
        'label': labels,
        'metadata': metadata_list
    }


def compute_loss(y, target, loss_mask, h, w_rnn, params):

    y_flat = y.reshape(-1, y.size(-1))
    target_flat = target.reshape(-1, target.size(-1))

    target_classes = torch.argmax(target_flat, dim=-1)
    ce_loss = nn.functional.cross_entropy(
        y_flat,
        target_classes,
        reduction='none'
    )
    ce_loss = ce_loss.reshape(y.size(0), y.size(1))
    perf_loss = torch.mean(ce_loss * loss_mask)

    spike_loss = torch.mean(h**2)
    weight_loss = torch.mean(w_rnn**2)

    spike_cost = params['spike_penalty']
    weight_cost = params['weight_penalty']
    perf_cost = params['perf_penalty']

    total_loss = perf_loss * perf_cost + spike_cost * spike_loss + weight_cost * weight_loss 

    return total_loss, perf_loss, spike_loss, weight_loss


def compute_accuracy(y, target, loss_mask, params):
    y_pred = torch.argmax(y, dim=-1)  # (T, B)
    target_class = torch.argmax(target, dim=-1)  # (T, B)

    # print('y_pred')
    # print(y_pred.shape)
    # print(y_pred[-6:])
    # print('target_class')
    # print(target_class[-6:])

    response_mask = (loss_mask == params['res_loss_mask'])  # (T, B)
    correct = (y_pred == target_class)
    # response_accuracy = correct[response_mask].float().mean().item()
    response_acc_count = 0
    for b in range(loss_mask.shape[1]):
        resp_idx = (response_mask[:, b]).nonzero(as_tuple=True)[0]
        if len(resp_idx) == 0:
            continue
        if correct[resp_idx, b].all():
            response_acc_count += 1
    response_accuracy = response_acc_count / loss_mask.shape[1]

    nonresponse_mask = (loss_mask == params['non_res_loss_mask'])  # (T, B)
    nonresponse_acc_count = 0
    for b in range(loss_mask.shape[1]):
        nonresp_idx = (nonresponse_mask[:, b]).nonzero(as_tuple=True)[0]
        if len(nonresp_idx) == 0:
            continue
        if correct[nonresp_idx, b].all():
            nonresponse_acc_count += 1
    nonresponse_accuracy = nonresponse_acc_count / loss_mask.shape[1]

    return response_accuracy, nonresponse_accuracy


def train_step(model, optimizer, inputs, targets, loss_mask, device, clip_grad, params, metadata_list=None):
    model.train()
    optimizer.zero_grad()

    if params['use_stsp']:
        hidden, syn_x, syn_u, y = model(inputs.to(device))
    else:
        hidden, y = model(inputs.to(device))

    w_rnn = model.h2h.weight

    loss, perf_loss, spike_loss, weight_loss = compute_loss(
        y, targets, loss_mask, hidden, w_rnn, params
    )

    response_accuracy, nonresponse_accuracy = compute_accuracy(y, targets, loss_mask, params)

    y_pred = torch.argmax(y, dim=-1)
    targets_class = torch.argmax(targets, dim=-1)
    response_mask = (loss_mask == params['res_loss_mask'])
    correct = (y_pred == targets_class)

    B = targets.shape[1]
    correct_mask = np.zeros(B, dtype=bool)
    for b in range(B):
        resp_idx = (response_mask[:, b]).nonzero(as_tuple=True)[0]
        if len(resp_idx) > 0:
            correct_mask[b] = correct[resp_idx, b].all().item()

    loss.backward()

    if params['w_rnn_mask'] is not None:
        model.h2h.weight.grad *= params['w_rnn_mask'].to(device)
    if params['w_output_mask'] is not None:
        model.h2output.weight.grad *= params['w_output_mask'].to(device)
    if params['w_input_mask'] is not None:
        model.input2h.weight.grad *= params['w_input_mask'].to(device)

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()

    return (
        loss.item(),
        perf_loss.item(),
        spike_loss.item(),
        weight_loss.item(),
        response_accuracy, 
        nonresponse_accuracy,
        correct_mask,
    )
    

# def evaluate(model, dataloader, device, PARAMS):
#     model.eval()

#     total_loss = 0.0
#     total_response_accuracy = 0.0
#     total_nonresponse_accuracy = 0.0
#     total_batches = 0

#     with torch.no_grad():
#         for batch in dataloader:
#             inputs = batch['input'].to(device)
#             targets = batch['label'].to(device)
#             metadata_list = batch['metadata']
#             B, T = targets.shape[0], targets.shape[1]
#             # Build loss mask: 1 everywhere, 2 at response_index, 0 at grace period
#             loss_mask = torch.ones(B, T, device=device)
#             for b in range(B):
#                 response_index = metadata_list[b]['response_index']
#                 if isinstance(response_index, torch.Tensor):
#                     response_index = response_index.cpu().numpy()
#                 response_index = np.array(response_index, dtype=int)
#                 grace_steps = PARAMS['grace_steps']
#                 grace_mask = response_index[:grace_steps]
#                 loss_mask[b, response_index] = 2
#                 loss_mask[b, grace_mask] = 0

#             inputs = inputs.transpose(0, 1)
#             targets = targets.transpose(0, 1)
#             loss_mask = loss_mask.transpose(0, 1)

#             if PARAMS['use_stsp']:
#                 hidden, syn_x, syn_u, y = model(inputs)
#             else:
#                 hidden, y = model(inputs)

#             w_rnn = model.h2h.weight

#             loss, perf_loss, spike_loss, weight_loss = compute_loss(
#                 y, targets, loss_mask, hidden, w_rnn, PARAMS
#             )

#             response_accuracy, nonresponse_accuracy = compute_accuracy(y, targets, loss_mask, PARAMS)

#             total_loss += loss.item()
#             total_response_accuracy += response_accuracy
#             total_nonresponse_accuracy += nonresponse_accuracy
#             total_batches += 1

#     avg_loss = total_loss / total_batches
#     avg_response_accuracy = total_response_accuracy / total_batches
#     avg_nonresponse_accuracy = total_nonresponse_accuracy / total_batches

#     return avg_loss, avg_response_accuracy, avg_nonresponse_accuracy


def train(
        model,
        save_dir=None,
        num_workers=1,
        verbose=False,
        **params,
):
    
    train_size = params['train_size']
    test_size = params['test_size']
    batch_size = params['batch_size']
    filter_conditions = params['filter_conditions']
    rhythmicity_pair_conditions = params['rhythmicity_pair_conditions']
    stimulus_num = params['stimulus_num']
    device = params['device']
    num_epochs = params['num_epochs']
    learning_rate = params['learning_rate']
    save_interval = params['save_interval']
    clip_grad = params['clip_grad']
    use_all_pairs = params['use_all_pairs']

    os.makedirs(save_dir, exist_ok=True)

    params_to_save = params.copy()
    for key, value in params_to_save.items():
        if isinstance(value, torch.Tensor):
            params_to_save[key] = value.tolist() if value.numel() < 100 else f"Tensor shape: {value.shape}"
        elif isinstance(value, np.ndarray):
            params_to_save[key] = value.tolist() if value.size < 100 else f"Array shape: {value.shape}"
        elif isinstance(value, dict) and key == 'filter_conditions':
            params_to_save[key] = f"List of {len(value)} filter condition dicts"
        elif callable(value):
            params_to_save[key] = f"<function {value.__name__}>"
        else:
            params_to_save[key] = str(value)
    
    with open(os.path.join(save_dir, 'parameters.json'), 'w') as f:
        json.dump(params_to_save, f, indent=4)
    print(f"Parameters saved to {os.path.join(save_dir, 'Parameters.json')}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    change_nums = params.get('change_nums', None)
    if isinstance(change_nums, np.ndarray):
        change_nums = change_nums.tolist()
    change_nums = sorted(list(set(change_nums)))


    history = {
        'train_loss': [],
        'train_response_accuracy': [],
        'train_nonresponse_accuracy': [],
        'train_perf_loss': [],
        'train_spike_loss': [],
        'train_weight_loss': [],
        'val_loss': [],
        'val_response_accuracy': [],
        'val_nonresponse_accuracy': [],
        'epoch': [],
        'train_rhythmic_accuracy': [],
        'train_arrhythmic_accuracy': [],
    }

    for change_num in change_nums:
        history[f'train_rhythmic_change_{change_num}_accuracy'] = []
        history[f'train_arrhythmic_change_{change_num}_accuracy'] = []

    best_train_response_accuracy = 0.0
    largest_rhy_arr_acc_diff = -1.0
    num_large_rhy_arr_acc_diff = 0

    datasets = create_datasets(**params)

    for filter_condition in filter_conditions:
        datasets.dataset_generator.channel_sequence_pair_database.statistics(**filter_condition)

    instances_to_save = dict()
    instances_to_save['datasets'] = datasets

    print(f"instance will save with test_variations = {datasets.dataset_generator.onset_generator.test_variations}.")

    with open(os.path.join(save_dir, 'instances.pkl'), 'wb') as f:
        pickle.dump(instances_to_save, f)
    print(" ")
    print(f"Instances saved to {os.path.join(save_dir, 'instances.pkl')}")

    print(" ")
    print("---------------------------------------------")
    print('Starting training...')
    print("---------------------------------------------")
    print(" ")
    start_time = time.time()

    os.makedirs(save_dir + "/model_params", exist_ok=True)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        current_use_all_pairs = params.get('use_all_pairs', False)

        train_dataset, test_dataset = datasets.create(
            train_size=train_size,
            test_size=test_size,
            filter_conditions=filter_conditions,
            rhythmicity_pair_conditions=rhythmicity_pair_conditions,
            device='cpu',
            use_all_pairs=current_use_all_pairs,
        )
        
        # print(f"Generated {len(train_dataset)} training trials and {len(test_dataset)} testing trials.")

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
            collate_fn=custom_collate_fn,
        )
        
        # val_loader = DataLoader(
        #     test_dataset,
        #     batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True,
        #     collate_fn=custom_collate_fn,
        # )
        
        epoch_loss = 0.0
        epoch_response_accuracy = 0.0
        epoch_nonresponse_accuracy = 0.0
        epoch_perf_loss = 0.0
        epoch_spike_loss = 0.0
        epoch_weight_loss = 0.0
        num_batches = 0

        
        accuracy_counts = {
            'rhythmic': {c: [0, 0] for c in change_nums},
            'arrhythmic': {c: [0, 0] for c in change_nums},
        }


        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            targets = batch['label'].to(device)
            metadata_list = batch['metadata']
            B, T = targets.shape[0], targets.shape[1]
            # loss_mask = torch.ones(B, T, device=device) 
            loss_mask = torch.full((B, T), params['non_res_loss_mask'], device=device)
            for b in range(B):
                response_index = metadata_list[b]['response_index']
                if isinstance(response_index, torch.Tensor):
                    response_index = response_index.cpu().numpy()
                response_index = np.array(response_index, dtype=int)
                grace_steps = params['grace_steps']
                grace_mask = response_index[:grace_steps]
                loss_mask[b, response_index] = params['res_loss_mask']
                loss_mask[b, grace_mask] = 0
                # print("loss_mask shape:", loss_mask.shape)
                # print(loss_mask[b])

            inputs = inputs.transpose(0, 1)
            targets = targets.transpose(0, 1)
            loss_mask = loss_mask.transpose(0, 1)

            loss, perf_loss, spike_loss, weight_loss, response_accuracy, nonresponse_accuracy, correct_mask = train_step(
                model, optimizer, inputs, targets, loss_mask, 
                device=device, clip_grad=clip_grad, params=params, metadata_list=metadata_list
            )


            for b in range(B):
                meta = metadata_list[b]
                sample_rhythmicity = meta.get('sample_rhythmicity', None)
                change_num = meta.get('change_num', None)

                if sample_rhythmicity is None or change_num is None:
                    print(f"Warning: Missing metadata for sample {b}")
                    continue

                if sample_rhythmicity in accuracy_counts and change_num in accuracy_counts[sample_rhythmicity]:
                    accuracy_counts[sample_rhythmicity][change_num][1] += 1
                    if correct_mask[b]:
                        accuracy_counts[sample_rhythmicity][change_num][0] += 1


            epoch_loss += loss
            epoch_response_accuracy += response_accuracy
            epoch_nonresponse_accuracy += nonresponse_accuracy
            epoch_perf_loss += perf_loss
            epoch_spike_loss += spike_loss
            epoch_weight_loss += weight_loss
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        avg_train_response_accuracy = epoch_response_accuracy / num_batches
        avg_train_nonresponse_accuracy = epoch_nonresponse_accuracy / num_batches
        avg_train_perf_loss = epoch_perf_loss / num_batches
        avg_train_spike_loss = epoch_spike_loss / num_batches
        avg_train_weight_loss = epoch_weight_loss / num_batches


        rhythmic_total_correct = sum(accuracy_counts['rhythmic'][c][0] for c in change_nums)
        rhythmic_total_count = sum(accuracy_counts['rhythmic'][c][1] for c in change_nums)
        rhythmic_acc = rhythmic_total_correct / rhythmic_total_count if rhythmic_total_count > 0 else 0.0

        arrhythmic_total_correct = sum(accuracy_counts['arrhythmic'][c][0] for c in change_nums)
        arrhythmic_total_count = sum(accuracy_counts['arrhythmic'][c][1] for c in change_nums)
        arrhythmic_acc = arrhythmic_total_correct / arrhythmic_total_count if arrhythmic_total_count > 0 else 0.0

        # val_loss, val_response_accuracy, val_nonresponse_accuracy = evaluate(model, val_loader, device, PARAMS)

        scheduler.step()

        history['train_loss'].append(avg_train_loss)
        history['train_response_accuracy'].append(avg_train_response_accuracy)
        history['train_nonresponse_accuracy'].append(avg_train_nonresponse_accuracy)
        history['train_perf_loss'].append(avg_train_perf_loss)
        history['train_spike_loss'].append(avg_train_spike_loss)
        history['train_weight_loss'].append(avg_train_weight_loss)
        history['train_rhythmic_accuracy'].append(rhythmic_acc)
        history['train_arrhythmic_accuracy'].append(arrhythmic_acc)
        # history['val_loss'].append(val_loss)
        # history['val_response_accuracy'].append(val_response_accuracy)
        # history['val_nonresponse_accuracy'].append(val_nonresponse_accuracy)
        history['epoch'].append(epoch + 1)


        for change_num in change_nums:
            rhythmic_change_acc = (accuracy_counts['rhythmic'][change_num][0] / accuracy_counts['rhythmic'][change_num][1]) \
                if accuracy_counts['rhythmic'][change_num][1] > 0 else 0.0
            arrhythmic_change_acc = (accuracy_counts['arrhythmic'][change_num][0] / accuracy_counts['arrhythmic'][change_num][1]) \
                if accuracy_counts['arrhythmic'][change_num][1] > 0 else 0.0
            
            history[f'train_rhythmic_change_{change_num}_accuracy'].append(rhythmic_change_acc)
            history[f'train_arrhythmic_change_{change_num}_accuracy'].append(arrhythmic_change_acc)


        epoch_time = time.time() - epoch_start_time


        rhythmic_change_str = ' | '.join([f'Rhythmic_Change{c}  : {history[f"train_rhythmic_change_{c}_accuracy"][-1]:.4f}' for c in change_nums])
        arrhythmic_change_str = ' | '.join([f'Arrhythmic_Change{c}: {history[f"train_arrhythmic_change_{c}_accuracy"][-1]:.4f}' for c in change_nums])


        print(f'=== Epoch [{epoch+1:3d}/{num_epochs}]  ( T {epoch_time:.2f}s ) ===')
        print(f'| Train Loss: {avg_train_loss:.4f} | Train response accuracy: {avg_train_response_accuracy:.4f} | Train nonresponse accuracy: {avg_train_nonresponse_accuracy:.4f} | ')
        print(f'    |   Rhythmic Acc: {rhythmic_acc:.4f} | {rhythmic_change_str} |')
        print(f'    | Arrhythmic Acc: {arrhythmic_acc:.4f} | {arrhythmic_change_str} |')
        # print(f'| Val Loss: {val_loss:.4f}   | Val response accuracy: {val_response_accuracy:.4f}   | Val nonresponse accuracy: {val_nonresponse_accuracy:.4f}   |')
        print(f'| Perf Loss: {avg_train_perf_loss:.4f} | Spike Loss: {avg_train_spike_loss:.4f} | Weight Loss: {avg_train_weight_loss:.4f} |')

        if avg_train_response_accuracy > best_train_response_accuracy:
            best_train_response_accuracy = avg_train_response_accuracy
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'train_response_accuracy': avg_train_response_accuracy,
        #         'params': params
        #     }, os.path.join(save_dir + "/model_params", 'best_model.pth'))
        #     print(f'Best model saved with train accuracy: {best_train_response_accuracy:.4f}')

        rhythmic_arrhythmic_diff = rhythmic_acc - arrhythmic_acc

        if avg_train_response_accuracy > 0.85:
            if rhythmic_arrhythmic_diff > largest_rhy_arr_acc_diff:
                largest_rhy_arr_acc_diff = rhythmic_arrhythmic_diff
            if (rhythmic_arrhythmic_diff >= 0.015) and (rhythmic_arrhythmic_diff >= 0.5 * largest_rhy_arr_acc_diff):
                num_large_rhy_arr_acc_diff += 1
                if (num_large_rhy_arr_acc_diff <= 100) or (rhythmic_arrhythmic_diff == largest_rhy_arr_acc_diff):
                    print("New large rhythmic-arrhythmic accuracy difference model saved!")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_response_accuracy': avg_train_response_accuracy,
                        'rhythmic_acc': rhythmic_acc,
                        'arrhythmic_acc': arrhythmic_acc,
                        'rhythmic_arrhythmic_diff': rhythmic_arrhythmic_diff,
                        'params': params,
                    }, os.path.join(save_dir + "/model_params", f'large_diff_model-epoch_{epoch+1}.pth'))

        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_response_accuracy': avg_train_response_accuracy,
                'rhythmic_acc': rhythmic_acc,
                'arrhythmic_acc': arrhythmic_acc,
                'rhythmic_arrhythmic_diff': rhythmic_arrhythmic_diff,
                'params': params,
            }, os.path.join(save_dir + "/model_params", f'checkpoint_epoch_{epoch+1}.pth'))
            print(f'Checkpoint saved at epoch {epoch+1}')

        total_time = time.time() - start_time
        # print('---------------------------------------------')
        print(f'Total training time so far: {total_time/60:.2f} minutes')
        print(f'Best training response accuracy so far: {best_train_response_accuracy:.4f}')
        # print('---------------------------------------------')

        with open(os.path.join(save_dir, 'training_history.txt'), 'a') as f:
            f.write(f"Epoch {history['epoch'][-1]}:\n")
            f.write(f"  Train Loss: {history['train_loss'][-1]:.6f}\n")
            f.write(f"  Train Response Accuracy: {history['train_response_accuracy'][-1]:.6f}\n")
            f.write(f"  Train Nonresponse Accuracy: {history['train_nonresponse_accuracy'][-1]:.6f}\n")

            f.write(f"  Train Rhythmic Accuracy: {history['train_rhythmic_accuracy'][-1]:.6f}\n")
            f.write(f"  Train Arrhythmic Accuracy: {history['train_arrhythmic_accuracy'][-1]:.6f}\n")
            for change_num in change_nums:
                f.write(f"    Rhythmic Change_{change_num}: {history[f'train_rhythmic_change_{change_num}_accuracy'][-1]:.6f}\n")
                f.write(f"    Arrhythmic Change_{change_num}: {history[f'train_arrhythmic_change_{change_num}_accuracy'][-1]:.6f}\n")

            f.write(f"  Perf Loss: {history['train_perf_loss'][-1]:.6f}\n")
            f.write(f"  Spike Loss: {history['train_spike_loss'][-1]:.6f}\n")
            f.write(f"  Weight Loss: {history['train_weight_loss'][-1]:.6f}\n")
            f.write("\n")

        print(" ")

    with open(os.path.join(save_dir, 'training_history.txt'), 'a') as f:
        f.write(f"Best Train Response Accuracy: {max(history['train_response_accuracy']):.6f}\n")
        
    return history


def plot_training_curves(history, save_dir):
    pass

