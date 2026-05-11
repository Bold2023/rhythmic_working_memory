
import argparse

import time

import random
import numpy as np
import torch

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from lib.parameters import TRAINING_PARAMS, DATASET_PARAMS, MODEL_PARAMS, DEVICE
from lib.initialize import initialize_PARAMS
from lib.EIRNN import EIRNN
from lib.train import train

from utils.typesetter import list_of_vals


NUM_WORKERS = 4
PARAMS = {**TRAINING_PARAMS, **DATASET_PARAMS, **MODEL_PARAMS}
PARAMS['device'] = DEVICE

parser = argparse.ArgumentParser(description=None)
for key, value in PARAMS.items():
    if type(value) is bool:
        parser.add_argument(f'--{key}', action='store_true')
    elif type(value) is list:
        parser.add_argument(f'--{key}', type=list_of_vals(value[0]), default=value)
    else:
        parser.add_argument(f'--{key}', type=type(value), default=value)
parser.add_argument('--identity', type=str, default='default', help='Identity of the run, used for saving models')
parser.add_argument('--time', type=str, default='000000000000', help='Time of the run, used for saving models')
args = parser.parse_args()

def main(args=args):

    print("Training with the following parameters:")

    params = vars(args)
    for key, value in params.items():
        print(f"    {key}: {value}")
    print(" ")

    params = initialize_PARAMS(params, DEVICE)
    
    stimulus_num = params['stimulus_num']
    save_parent_dir = params['save_parent_dir']

    test_variations = params.get('test_variations', None)
    tv_str = "-tvu" if test_variations is None else ""

    use_all_pairs = params.get('use_all_pairs', False)
    uap_str = "-uap" if use_all_pairs else ""

    device = DEVICE
    print(f'Using device: {device}')

    print(" ")

    save_dir = save_parent_dir + \
        f"stim_{stimulus_num}" + \
        "/" + \
        \
        f"change_{'_'.join(map(str, params['change_nums']))}" + \
        "-" + f"m_{params['tuning_neuron_num']}_{params['hidden_size']}_{params['output_size']}" + \
        "/" + \
        \
        f"EIRNN_" + ("STSP" if params['use_stsp'] else "vanilla") + \
        uap_str + \
        "-" + f"dur_{params['stimulus_duration']}" + \
        "-" + f"ioi_{params['inter_onset_interval']}" + \
        "-" + f"delay_{params['delay_duration']}" + \
        tv_str + \
        "/" + \
        \
        f"time_{params['time']}" + \
        "-" + f"turn_{params['identity'].zfill(2)}"

    print(f'Saving models to: {save_dir}')

    num_workers = NUM_WORKERS

    model = EIRNN(**params).to(device)

    history = train(
        model=model,
        save_dir=save_dir,
        num_workers=num_workers,
        **params,
    )


if __name__ == '__main__':

    print("starting main.py")
    main()
    print("main.py finished")
