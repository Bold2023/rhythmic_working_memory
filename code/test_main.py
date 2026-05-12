import os
import json
from pathlib import Path
from datetime import datetime
import re
from lib.test import set_global_seed, test_model
import pandas as pd 


SEED_LIST = [42]
TEST_SIZE = 2400
# MODEL_PATH = './models/stim_4/251128-c_0_2-m_32_100_3/turn_19/model_params/checkpoint_epoch_4000.pth'
OUTPUT_DIR = './STSP_RNN_stim6results_30-100_tvu/140-500-1000'
SAVE_ACTIVATIONS = False
SHARE_ARRHYTHMIC_SAMPLE_ONSETS = True
SHARE_CH3_TEST_SEQS = True
SHUFFLE_TIME_STEP = None

def parse_model_info(model_path: str, seed: int):
    p = Path(model_path)
    model_dir_name = p.parent.parent.name
    m_epoch = re.search(r'epoch_(\d+)', p.name)
    epoch = int(m_epoch.group(1)) if m_epoch else None
    m_seed_dir = re.search(r'seed_(\d+)', model_dir_name)
    seed_in_dir = int(m_seed_dir.group(1)) if m_seed_dir else None
    final_seed = seed if seed_in_dir is None or seed_in_dir != seed else seed_in_dir
    return {
        'model_dir_name': model_dir_name,
        'checkpoint_filename': p.name,
        'epoch': epoch,
        'seed': final_seed,
        'abs_checkpoint_path': str(p.resolve())
    }

def main():
    print('Start testing...')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for turn in range(1,11):
        MODEL_PATH = f'./stim_6_1219_30-100-tvu/EIRNN_STSP-uap-dur_140-ioi_500-delay_1000-tvu/time_202512231411-turn_{turn:02d}/model_params/checkpoint_epoch_5000.pth'
        if not os.path.exists(MODEL_PATH):
            print(f'Model not found: {MODEL_PATH}')
            continue

        all_results = []
        # run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        for seed in SEED_LIST:
            print(f'Use seed: {seed}')
            set_global_seed(seed)

            info = parse_model_info(MODEL_PATH, seed)
            test_result = test_model(MODEL_PATH, test_size=TEST_SIZE, 
                                    save_activations=SAVE_ACTIVATIONS,
                                    share_arrhythmic_sample_onsets=SHARE_ARRHYTHMIC_SAMPLE_ONSETS,
                                    share_ch3_test_seqs=SHARE_CH3_TEST_SEQS,
                                    use_stsp=None,
                                    shuffle_time_step=SHUFFLE_TIME_STEP
                                    )

            all_results.append({
                'meta': {
                    **info,
                    'test_size': TEST_SIZE,
                    'run_timestamp': run_timestamp
                },
                'results': test_result
            })

        summary_name = f"{run_timestamp}_{Path(MODEL_PATH).parent.parent.name}_epoch{all_results[0]['meta']['epoch']}.json"
        summary_path = Path(OUTPUT_DIR) / summary_name

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({'all_seed_results': all_results}, f, ensure_ascii=False, indent=2)

        print(f'Saved summary: {summary_path}')

if __name__ == '__main__':
    main()
