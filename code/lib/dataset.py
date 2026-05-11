# from utils.save_err_out import save_system_message
# save_system_message(False)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import random



VERSION_STRING = "final"



class TimelineCreator():
    def __init__(self, **params):  

        self.stimulus_num = params['stimulus_num']
        self.channel_num = params['channel_num']
        self.direction_num = params['direction_num']
        self.channel_to_direction_mapping = params.get('channel_to_direction_mapping', None)
        if self.channel_to_direction_mapping is None:
            self.channel_to_direction_mapping = list(range(self.channel_num))

        assert len(self.channel_to_direction_mapping) == self.channel_num, "Length of channel_to_direction_mapping must equal channel_num."

        self.fixation_duration = params['fixation_duration']
        self.delay_duration = params['delay_duration']
        self.response_duration = params['response_duration']
        self.stimulus_duration = params['stimulus_duration']
        self.inter_onset_interval = params['inter_onset_interval']
        self.time_step = params['time_step']

        assert (self.fixation_duration % self.time_step) == 0, "fixation_duration must be divisible by time_step."
        assert (self.delay_duration % self.time_step) == 0, "delay_duration must be divisible by time_step."
        assert (self.response_duration % self.time_step) == 0, "response_duration must be divisible by time_step."
        assert (self.stimulus_duration % self.time_step) == 0, "stimulus_duration must be divisible by time_step."
        assert (self.inter_onset_interval % self.time_step) == 0, "inter_onset_interval must be divisible by time_step."

        self.sample_duration = self.inter_onset_interval * (self.stimulus_num - 1) + self.stimulus_duration
        self.test_duration = self.inter_onset_interval * (self.stimulus_num - 1) + self.stimulus_duration

        self.fixation_duration_steps = self.fixation_duration // self.time_step
        self.sample_duration_steps = self.sample_duration // self.time_step
        self.delay_duration_steps = self.delay_duration // self.time_step
        self.test_duration_steps = self.test_duration // self.time_step
        self.response_duration_steps = self.response_duration // self.time_step
        self.stimulus_duration_steps = self.stimulus_duration // self.time_step

        self.tuning_neuron_num = params.get('tuning_neuron_num', 30)
        self.tuning_height = params.get('tuning_height', 4)
        self.tuning_kappa = params.get('tuning_kappa', 2)

        self.tuning_function_init()

    def place_sample_test_stimuluses(self, sample_seq, test_seq, sample_onsets, test_onsets):
        sample_block = np.zeros((self.sample_duration_steps, self.direction_num))
        test_block = np.zeros((self.test_duration_steps, self.direction_num))

        for _, (sample_onset, sample_channel) in enumerate(zip(sample_onsets, sample_seq)):
            sample_block[sample_onset:sample_onset+self.stimulus_duration_steps, self.channel_to_direction_mapping[sample_channel]] = 1

        for _, (test_onset, test_channel) in enumerate(zip(test_onsets, test_seq)):
            test_block[test_onset:test_onset+self.stimulus_duration_steps, self.channel_to_direction_mapping[test_channel]] = 1

        return sample_block, test_block

    def tuning_function_init(self):
        direction_rotation = np.linspace(0, 2*np.pi, self.direction_num, endpoint=False)
        preferred_rotation = np.linspace(0, 2*np.pi, self.tuning_neuron_num, endpoint=False)

        self.tuning_function_matrix = np.zeros((self.direction_num, self.tuning_neuron_num))

        for p in range(self.direction_num):
            for n in range(self.tuning_neuron_num):
                dist = np.cos(direction_rotation[p] - preferred_rotation[n])
                self.tuning_function_matrix[p, n] = self.tuning_height * np.exp(self.tuning_kappa * dist) / np.exp(self.tuning_kappa)

    def tune_sample_test_blocks(self, sample_block, test_block):
        tuned_sample_block = sample_block @ self.tuning_function_matrix
        tuned_test_block = test_block @ self.tuning_function_matrix
        return tuned_sample_block, tuned_test_block
    
    def create_timeline(self, sample_seq, test_seq, sample_onsets, test_onsets, label, cue_neuron=True):
        sample_block, test_block = self.place_sample_test_stimuluses(sample_seq, test_seq, sample_onsets, test_onsets)
        tuned_sample_block, tuned_test_block = self.tune_sample_test_blocks(sample_block, test_block)

        timeline = np.concatenate([
            np.zeros((self.fixation_duration_steps, self.tuning_neuron_num)),
            tuned_sample_block,
            np.zeros((self.delay_duration_steps, self.tuning_neuron_num)),
            tuned_test_block,
            np.zeros((self.response_duration_steps, self.tuning_neuron_num)),
        ], axis=0)

        if cue_neuron:
            cue = np.ones((timeline.shape[0], 1))
            cue[-self.response_duration_steps:, 0] = 0
            timeline = np.concatenate([timeline, cue], axis=-1)

        label_timeline = np.zeros((timeline.shape[0], 3))
        label_timeline[:-self.response_duration_steps, 0] = 1
        if label == "match":
            label_timeline[-self.response_duration_steps:, 1] = 1
        elif label == "nonmatch":
            label_timeline[-self.response_duration_steps:, 2] = 1

        return timeline, label_timeline


def filter_dicts_by_keywords(dicts, keyword_val_pairs=[], filter_mode="equal"):
        '''
            dicts: (list of dict) List of dicts to filter.
            keyword_val_pairs: (list of tuples) Each tuple contains (keyword, keyword_vals).
                keyword: (str) The key in the seq_pair dictionary to filter by.
                keyword_vals: (any or list of any) The value(s) to filter by.
            filter_mode: (str = "equal") Mode of filtering. Can be "equal" or "include".
        '''
        for keyword, keyword_vals in keyword_val_pairs:
            if filter_mode == "equal":
                dicts = filter(lambda seq_pair: seq_pair.get(keyword) == keyword_vals, dicts)
            elif filter_mode == "include":
                dicts = filter(lambda seq_pair: seq_pair.get(keyword) in keyword_vals, dicts)
        return list(dicts)

def filter_dicts_by_tags(dicts, include_tags=None, exclude_tags=None):
        '''
            dicts: (list of dict) List of dicts to filter.
            include_tags: (list of str = None) Tags that must be included. If None, no inclusion filtering is applied.
            exclude_tags: (list of str = None) Tags that must be excluded. If None, no exclusion filtering is applied.
        '''
        if include_tags:
            for tag in include_tags:
                dicts = filter(lambda seq_pair: tag in seq_pair["tags"], dicts)
        if exclude_tags:
            for tag in exclude_tags:
                dicts = filter(lambda seq_pair: tag not in seq_pair["tags"], dicts)
        return list(dicts)

class ChannelSequencePairDatabase():
    def __init__(self,
            **params,
        ):
        '''
            stimulus_num: (int) Number of stimuluses in each sequence.
            channel_num: (int = 7) Number of input channels.
            change_nums: (list of int = []) Number of changes for each trial.
            allow_repeat: (bool = False) Whether to allow repeats within a sequence.
            permitted_movements: (list of shape (N, 2)) Permitted movements between channels.
        '''
        self.stimulus_num = params['stimulus_num']
        self.channel_num = params['channel_num']
        self.change_nums = self.verify_change_nums(params['change_nums'])
        self.allow_repeat = params.get('allow_repeat', False)
        self.permitted_movements = params.get('permitted_movements', None)
        self.permitted_movement_directions = params.get('permitted_movement_directions', ["positive", "negative"])

        sample_seqs = self.generate_sample_sequences()

        seq_pairs = []
        for sample_seq in sample_seqs:
            for change_num in self.change_nums:
                seq_pair_for_sample = self.generate_sequence_pair(sample_seq, change_num)
                seq_pairs.extend(seq_pair_for_sample)
        self.seq_pairs = seq_pairs

        if not self.allow_repeat:
            self.filter_by_tags(exclude_tags=["sample_repeat", "test_repeat"], in_place=True)

        self.statistics()

    def verify_change_nums(self, change_nums):
        sorted_changes = sorted(list(set(change_nums)))    # Remove duplicates and sort
        if not all(isinstance(num, int) for num in sorted_changes):
            raise ValueError("all change_nums must be integers.")
        if not all(num >= 0 for num in sorted_changes):
            raise ValueError("all change_nums must be non-negative.")
        if 0 not in sorted_changes:
            sorted_changes.insert(0, 0)
        if sorted_changes[-1] > self.stimulus_num:
            raise ValueError(f"all change_nums must be no more than stimulus_num ({self.stimulus_num}).")

        for change_num in sorted_changes:
            if change_num != 0:
                if self.stimulus_num % change_num != 0:
                    print(f"Warning: change_num {change_num} is not a divisor of stimulus_num {self.stimulus_num}.")
                
        return sorted_changes

    def generate_sample_sequences(self):
        channel_list = np.array(list(range(self.channel_num)))
        sample_seqs = np.meshgrid(*([channel_list] * self.stimulus_num), indexing='ij')
        return [sample_seq for sample_seq in np.stack(sample_seqs, axis=-1).reshape(-1, self.stimulus_num)]

    def generate_sequence_pair(self, sample_seq, change_num):
        '''
            sample_seq: (array of int with shape (stimulus_num,)) Sample sequence.
            change_num: (int) Number of changes.
        '''

        if change_num == 0:
            tags = []
            if any(sample_seq[i] == sample_seq[i+1] for i in range(len(sample_seq)-1)):
                tags.append("sample_repeat")
            if any(sample_seq[i] == sample_seq[i+1] for i in range(len(sample_seq)-1)):
                tags.append("test_repeat")
            pair = (sample_seq, sample_seq.copy())
            pair_dict = {
                "seq_pair": pair,
                "change_num": change_num,
                "tags": tags,
            }
            return [pair_dict]

        seq_pairs = []

        stimuluses_per_segment = self.stimulus_num // change_num
        possible_pair_num = stimuluses_per_segment + self.stimulus_num % change_num

        for first_change_idx in range(possible_pair_num):   # change position
            change_idxs = [(first_change_idx + i * stimuluses_per_segment) for i in range(change_num)]

            for direction in self.permitted_movement_directions:      # change direction

                test_seq = sample_seq.copy()

                if direction == "positive":
                    if self.permitted_movements is not None:
                        if not all([test_seq[change_idx], (test_seq[change_idx] + 1) % self.channel_num] in self.permitted_movements for change_idx in change_idxs):
                            continue
                    for change_idx in change_idxs:
                        test_seq[change_idx] = (test_seq[change_idx] + 1) % self.channel_num
                elif direction == "negative":
                    if self.permitted_movements is not None:
                        if not all([test_seq[change_idx], (test_seq[change_idx] + self.channel_num - 1) % self.channel_num] in self.permitted_movements for change_idx in change_idxs):
                            continue
                    for change_idx in change_idxs:
                        test_seq[change_idx] = (test_seq[change_idx] + self.channel_num - 1) % self.channel_num

                tags = []
                if any(sample_seq[i] == sample_seq[i+1] for i in range(len(sample_seq)-1)):
                    tags.append("sample_repeat")
                if any(test_seq[i] == test_seq[i+1] for i in range(len(test_seq)-1)):
                    tags.append("test_repeat")
                pair = (sample_seq, test_seq)
                pair_dict = {
                    "seq_pair": pair,
                    "change_num": change_num,
                    "direction": direction, 
                    "change_idxs": change_idxs,
                    "tags": tags,
                }
                seq_pairs.append(pair_dict)

        return seq_pairs

    def filter_by_keywords(self, keyword_val_pairs=[], filter_mode="equal", in_place=False):
        '''
            keyword_val_pairs: (list of tuples) Each tuple contains (keyword, keyword_vals).
                keyword: (str) The key in the seq_pair dictionary to filter by.
                keyword_vals: (any or list of any) The value(s) to filter by.
            filter_mode: (str = "equal") Mode of filtering. Can be "equal" or "include".
            in_place: (bool) Whether to modify the seq_pairs in place.
        '''
        filtered_dicts = filter_dicts_by_keywords(self.seq_pairs, keyword_val_pairs=keyword_val_pairs, filter_mode=filter_mode)
        if in_place:
            self.seq_pairs = filtered_dicts
            return None
        return filtered_dicts

    def filter_by_tags(self, include_tags=None, exclude_tags=None, in_place=False):
        '''
            include_tags: (list of str = None) Tags that must be included. If None, no inclusion filtering is applied.
            exclude_tags: (list of str = None) Tags that must be excluded. If None, no exclusion filtering is applied.
            in_place: (bool) Whether to modify the seq_pairs in place.
        '''
        filtered_dicts = filter_dicts_by_tags(self.seq_pairs, include_tags=include_tags, exclude_tags=exclude_tags)
        if in_place:
            self.seq_pairs = filtered_dicts
            return None
        return filtered_dicts

    def statistics(self, keyword_val_pairs=[], filter_mode="equal", include_tags=None, exclude_tags=None):

        total_num = len(self.seq_pairs)
        filtered_pairs = self.seq_pairs

        if filter_mode == "equal":
            keyword_connector = "=="
        elif filter_mode == "include":
            keyword_connector = "in"

        flag = False
        for keyword, keyword_vals in keyword_val_pairs:
            flag = True
            filtered_pairs = filter_dicts_by_keywords(filtered_pairs, keyword_val_pairs=[(keyword, keyword_vals)], filter_mode=filter_mode)
        if include_tags:
            flag = True
            print(f"Filtering to include tags: {include_tags}")
            filtered_pairs = filter_dicts_by_tags(filtered_pairs, include_tags=include_tags)
        if exclude_tags:
            flag = True
            print(f"Filtering to exclude tags: {exclude_tags}")
            filtered_pairs = filter_dicts_by_tags(filtered_pairs, exclude_tags=exclude_tags)

        if flag:
            print(f"{len(filtered_pairs)} pairs for filter: <"
                  f"{', '.join([f'{k} {keyword_connector} {v}' for k, v in keyword_val_pairs])}, "
                  f"include_tags={include_tags}, exclude_tags={exclude_tags}"
                  f">")
        else:
            print(f"Total sequence pairs: {total_num}")

    def filtered_generator(self, keyword_val_pairs=[], filter_mode="equal", include_tags=None, exclude_tags=None, shuffle=True):
        filtered_pairs = self.seq_pairs

        for keyword, keyword_vals in keyword_val_pairs:
            filtered_pairs = filter_dicts_by_keywords(filtered_pairs, keyword_val_pairs=[(keyword, keyword_vals)], filter_mode=filter_mode)
        filtered_pairs = filter_dicts_by_tags(filtered_pairs, include_tags=include_tags, exclude_tags=exclude_tags)

        if shuffle:
            np.random.shuffle(filtered_pairs)

        for pair in filtered_pairs:
            yield pair


class OnsetGenerator():
    def __init__(self, 
            strict_rhythmic_hierarchy=True,
            freeze_test_rhythmicity=True,
            test_variations=None,
            **params,
        ):
            self.time_step = params['time_step']

            self.stimulus_num = params['stimulus_num']
            self.stimulus_duration = params['stimulus_duration']
            self.inter_onset_interval = params['inter_onset_interval']
            self.arrhythmic_ratio = params['arrhythmic_ratio']
            self.semirhythmic_ratio = params['semirhythmic_ratio']


            self.stimulus_duration_steps = self.stimulus_duration // self.time_step
            self.inter_onset_interval_steps = self.inter_onset_interval // self.time_step

            self.arrhythmic_max_deviation_steps = int(self.inter_onset_interval_steps * self.arrhythmic_ratio / 2)
            print(self.arrhythmic_max_deviation_steps, "is the arrhythmic max deviation in steps.")
            self.semirhythmic_max_deviation_steps = int(self.inter_onset_interval_steps * self.semirhythmic_ratio / 2)
            print(self.semirhythmic_max_deviation_steps, "is the semirhythmic max deviation in steps.")

            self.strict_rhythmic_hierarchy = strict_rhythmic_hierarchy
            self.freeze_test_rhythmicity = freeze_test_rhythmicity
            if self.freeze_test_rhythmicity:
                if test_variations is not None:
                    test_variations = np.array(test_variations)
                    abs_variation = np.absolute(test_variations)
                    assert (test_variations.shape[0] == self.stimulus_num - 2), f"test_variations must have length {self.stimulus_num - 2} (stimulus_num - 2)."
                    assert (np.all(abs_variation <= self.semirhythmic_max_deviation_steps)), "test_variations must be within the semirhythmic max deviation."
                    assert not(np.all(test_variations == 0)), "test_variations must be not all zero."
                    print(f"Test variations has been designated as {test_variations}.")
                    print(" ")
                else:
                    while True:
                        test_variations = np.random.randint(-self.semirhythmic_max_deviation_steps, self.semirhythmic_max_deviation_steps + 1, size=self.stimulus_num - 2)
                        if self.strict_rhythmic_hierarchy \
                        and all(abs_variation == 0 for abs_variation in np.absolute(test_variations)):
                            continue
                        else:
                            print(f"Test variations has been randomly generated as {test_variations}.")
                            print(" ")
                            break

                self.test_variations = test_variations

            assert (self.stimulus_duration % self.time_step) == 0, "stimulus_duration must be divisible by time_step."
            assert (self.inter_onset_interval % self.time_step) == 0, "inter_onset_interval must be divisible by time_step."

    def verify(self):   # TODO
        pass

    def generate_test_variations(self, test_rhythmicity="semirhythmic"):
        if test_rhythmicity == 'semirhythmic':
            if self.freeze_test_rhythmicity:
                return self.test_variations
            else:
                test_variations = np.random.randint(-self.semirhythmic_max_deviation_steps, self.semirhythmic_max_deviation_steps + 1, size=self.stimulus_num - 2)
                return test_variations
        else:
            raise ValueError(f"test_rhythmicity = {test_rhythmicity} not implemented.")

    def generate_onsets(self, sample_rhythmicity, test_rhythmicity='semirhythmic'):
        '''
            sample_rhythmicity: 'rhythmic' or 'arrhythmic'
            test_rhythmicity: 'semirhythmic'
        '''
        
        sample_onsets = np.array([stimulus_idx * self.inter_onset_interval_steps for stimulus_idx in range(self.stimulus_num)])
        test_onsets = sample_onsets.copy()

        if sample_rhythmicity == 'rhythmic':
            pass
        elif sample_rhythmicity == 'arrhythmic':
            while True:
                sample_variations = np.random.randint(-self.arrhythmic_max_deviation_steps, self.arrhythmic_max_deviation_steps + 1, size=self.stimulus_num - 2)
                if self.strict_rhythmic_hierarchy \
                and all(abs_variation < self.semirhythmic_max_deviation_steps for abs_variation in np.absolute(sample_variations)):
                    continue
                else:
                    break
            sample_onsets += np.concatenate(([0], sample_variations, [0]))

        elif sample_rhythmicity == 'random':
            sample_onsets = np.random.randint(0, self.inter_onset_interval_steps * (self.stimulus_num - 1), size=self.stimulus_num)
            sample_onsets.sort()
            sample_onsets[0] = 0
            sample_onsets[-1] = self.inter_onset_interval_steps * (self.stimulus_num - 1)
        else:
            raise ValueError(f"sample_rhythmicity = {sample_rhythmicity} not implemented.")

        test_variations = self.generate_test_variations(test_rhythmicity)
        test_onsets += np.concatenate(([0], test_variations, [0]))
        
        pair_dict = {
            "onset_pair": (sample_onsets, test_onsets),
            "sample_rhythmicity": sample_rhythmicity,
            "test_rhythmicity": test_rhythmicity,
        }

        return pair_dict


class DatasetGenerator():
    def __init__(self,
            timeline_creator: TimelineCreator, 
            channel_sequence_pair_database: ChannelSequencePairDatabase,
            onset_generator: OnsetGenerator,
            cue_neuron=True,
        ):
        self.timeline_creator = timeline_creator
        self.channel_sequence_pair_database = channel_sequence_pair_database
        self.onset_generator = onset_generator
        self.cue_neuron = cue_neuron

    def generate_trial_dict(self, channel_seq_pair, onset_pair):
        trial_dict = dict()
        trial_dict.update(channel_seq_pair)
        trial_dict.update(onset_pair)

        sample_seq, test_seq = trial_dict["seq_pair"]
        sample_onsets, test_onsets = trial_dict["onset_pair"]

        timeline, label_timeline = self.timeline_creator.create_timeline(
            sample_seq=sample_seq, test_seq=test_seq,
            sample_onsets=sample_onsets, test_onsets=test_onsets,
            label="match" if trial_dict["change_num"] == 0 else "nonmatch",
            cue_neuron=self.cue_neuron,
        )

        trial_dict["data"] = timeline
        trial_dict["label"] = label_timeline

        # response_index: indices (time steps) corresponding to the response period at the end of the timeline
        resp_len = self.timeline_creator.response_duration_steps
        T = timeline.shape[0]
        response_index = list(range(T - resp_len, T))
        trial_dict["response_index"] = response_index

        return trial_dict
    
    def generate_filtered_trial_dictsets(self,
            rhythmicity_pairs=[],
            keyword_val_pairs=[], filter_mode="equal", include_tags=None, exclude_tags=None, 
            shuffle=True,
        ):
        '''
            rhythmicity_pairs: (list of tuples) Each tuple contains (sample_rhythmicity, test_rhythmicity).
                sample_rhythmicity: 'rhythmic' or 'arrhythmic'
                test_rhythmicity: 'semirhythmic'
            keyword_val_pairs: (list of tuples) Each tuple contains (keyword, keyword_vals).
                keyword: (str) The key in the seq_pair dictionary to filter by.
                keyword_vals: (any or list of any) The value(s) to filter by.
            filter_mode: (str = "equal") Mode of filtering. Can be "equal" or "include".
            include_tags: (list of str = None) Tags that must be included. If None, no inclusion filtering is applied.
            exclude_tags: (list of str = None) Tags that must be excluded. If None, no exclusion filtering is applied.
            shuffle: (bool = True) Whether to shuffle the filtered sequence pairs before splitting.
        '''
        total_size = len(rhythmicity_pairs)

        filtered_seq_pairs = list(self.channel_sequence_pair_database.filtered_generator(
            keyword_val_pairs=keyword_val_pairs, filter_mode=filter_mode,
            include_tags=include_tags, exclude_tags=exclude_tags, shuffle=shuffle
        ))

        # print(f"Filtered sequence pairs: {len(filtered_seq_pairs)}")
        if len(filtered_seq_pairs) < total_size:
            raise ValueError(f"Not enough sequence pairs after filtering. Required: {total_size}, Available: {len(filtered_seq_pairs)}")

        trial_dicts = []
        for seq_pair_dict, (sample_rhythmicity, test_rhythmicity) in zip(filtered_seq_pairs[:total_size], rhythmicity_pairs):
            onset_pair_dict = self.onset_generator.generate_onsets(sample_rhythmicity=sample_rhythmicity, test_rhythmicity=test_rhythmicity)
            trial_dicts.append(self.generate_trial_dict(seq_pair_dict, onset_pair_dict))

        return trial_dicts

    def generate_datasets(self,
            train_size, test_size, shuffle=True,
            filter_conditions: list[dict] = None,
            rhythmicity_pair_conditions=[("rhythmic", "semirhythmic"), ("arrhythmic", "semirhythmic")],
            use_all_pairs=False,
        ):
        '''
            train_size: (int) Number of training trials.
            test_size: (int) Number of testing trials.
            shuffle: (bool = True) Whether to shuffle the filtered sequence pairs before splitting.
            filter_conditions: (list of dict = None) Each dict contains keyword_val_pairs, filter_mode, include_tags, exclude_tags for filtering. If None, no filtering is applied. Data will be evenly split among these conditions.
                keyword_val_pairs: (list of tuples) Each tuple contains (keyword, keyword_vals).
                    keyword: (str) The key in the seq_pair dictionary to filter by.
                    keyword_vals: (any or list of any) The value(s) to filter by.
                filter_mode: (str = "equal") Mode of filtering. Can be "equal" or "include".
                include_tags: (list of str = None) Tags that must be included. If None, no inclusion filtering is applied.
                exclude_tags: (list of str = None) Tags that must be excluded. If None, no exclusion filtering is applied.
            rhythmicity_pair_conditions: (list of tuples) Each tuple contains (sample_rhythmicity, test_rhythmicity). Data will be evenly split among these conditions.
                sample_rhythmicity: 'rhythmic' or 'arrhythmic'
                test_rhythmicity: 'semirhythmic'
        '''

        if filter_conditions is None:
            filter_conditions = [dict(
                keyword_val_pairs=[],
                filter_mode="equal",
                include_tags=None,
                exclude_tags=None,
            )]

        train_dicts = []
        test_dicts = []


###############################################新加超参#############################################
        if use_all_pairs:

            train_dicts_dict = dict()
            print("Generating exhaustive training dataset (covering all sequence pairs)...")

            for filter_condition_idx, filter_condition in enumerate(filter_conditions):
                train_dicts_dict[filter_condition_idx] = []
                pairs = list(self.channel_sequence_pair_database.filtered_generator(
                    **filter_condition, shuffle=shuffle
                ))
                print(f"Found {len(pairs)} pairs for condition: {filter_condition}")
                
                for pair in pairs:
                    for rhythm_pair in rhythmicity_pair_conditions:
                        sample_rhythmicity, test_rhythmicity = rhythm_pair
                        onset_pair = self.onset_generator.generate_onsets(sample_rhythmicity, test_rhythmicity)
                        trial = self.generate_trial_dict(pair, onset_pair)
                        train_dicts_dict[filter_condition_idx].append(trial)
            
            if test_size > 0:
                 assert test_size % (len(filter_conditions) * len(rhythmicity_pair_conditions)) == 0, "test_size must be divisible by len(filter_conditions) * len(rhythmicity_pair_conditions)."
                 test_size_per_fc_per_rc = test_size // (len(filter_conditions) * len(rhythmicity_pair_conditions))
                 
                 for filter_condition in filter_conditions:
                     test_rhythmicity_pairs = [rhythmicity_pair for rhythmicity_pair in rhythmicity_pair_conditions for _ in range(test_size_per_fc_per_rc)]
                     
                     filtered_dicts = self.generate_filtered_trial_dictsets(
                        rhythmicity_pairs=test_rhythmicity_pairs,
                        **filter_condition,
                        shuffle=shuffle,
                    )
                     test_dicts.extend(filtered_dicts)

            return train_dicts_dict, test_dicts
#####################################################################################################


        assert train_size % (len(filter_conditions) * len(rhythmicity_pair_conditions)) == 0, "train_size must be divisible by len(filter_conditions) * len(rhythmicity_pair_conditions)."
        assert test_size % (len(filter_conditions) * len(rhythmicity_pair_conditions)) == 0, "test_size must be divisible by len(filter_conditions) * len(rhythmicity_pair_conditions)."

        train_size_per_fc = train_size // len(filter_conditions)
        test_size_per_fc = test_size // len(filter_conditions)

        train_size_per_fc_per_rc = train_size // (len(filter_conditions) * len(rhythmicity_pair_conditions))
        test_size_per_fc_per_rc = test_size // (len(filter_conditions) * len(rhythmicity_pair_conditions))

        for filter_condition in filter_conditions:
            # print("Generating sub-dataset with filter condition:")
            # print(filter_condition)

            train_rhythmicity_pairs = [rhythmicity_pair for rhythmicity_pair in rhythmicity_pair_conditions for _ in range(train_size_per_fc_per_rc)]
            test_rhythmicity_pairs = [rhythmicity_pair for rhythmicity_pair in rhythmicity_pair_conditions for _ in range(test_size_per_fc_per_rc)]

            filtered_dicts = self.generate_filtered_trial_dictsets(
                rhythmicity_pairs=train_rhythmicity_pairs + test_rhythmicity_pairs,
                **filter_condition,
                shuffle=shuffle,
            )
            # print(f"Generated {len(filtered_dicts)} trials for this filter condition.")
            # print(" ")
            train_dicts.extend(filtered_dicts[:train_size_per_fc])
            test_dicts.extend(filtered_dicts[train_size_per_fc:])

        return train_dicts, test_dicts


class DMSDataset(Dataset):
    def __init__(self, trial_dicts, device='cpu'):
        self.trial_dicts = trial_dicts
        self.device = device

    def __len__(self):
        return len(self.trial_dicts)

    def __getitem__(self, idx):
        trial_dict = self.trial_dicts[idx]
        X = torch.tensor(trial_dict['data']).float().to(self.device)
        y = torch.tensor(trial_dict['label']).float().to(self.device)

        metadata = {k: v for k, v in trial_dict.items() if k not in ['data', 'label']}
        
        return {
            'input': X,
            'label': y, 
            'metadata': metadata,
        }


# change_0_condition = dict(
#     keyword_val_pairs=[("change_num", 0)],
#     filter_mode="equal",
#     include_tags=None,
#     exclude_tags=None,
# )
# change_1_condition = dict(
#     keyword_val_pairs=[("change_num", 1)],
#     filter_mode="equal",
#     include_tags=None,
#     exclude_tags=None,
# )
# change_2_condition = dict(
#     keyword_val_pairs=[("change_num", 2)],
#     filter_mode="equal",
#     include_tags=None,
#     exclude_tags=None,
# )


class create_datasets():

    def __init__(self, **params):

        print(" ")

        self.version = VERSION_STRING

        timeline_creator = TimelineCreator(**params)
        channel_sequence_pair_database = ChannelSequencePairDatabase(**params)
        onset_generator = OnsetGenerator(**params)

        self.dataset_generator = DatasetGenerator(
            timeline_creator=timeline_creator,
            channel_sequence_pair_database=channel_sequence_pair_database,
            onset_generator=onset_generator,
            cue_neuron=True,
        )

        self.cached_all_pairs_train_dicts = None
        self.cached_all_pairs_test_dicts = None
        self.current_index = 0

    def create(self,
        train_size, test_size,
        filter_conditions=None, rhythmicity_pair_conditions=None,
        device=torch.device('cpu'),
        use_all_pairs = False,
    ):
        if filter_conditions is None:
            filter_conditions = [dict(
                keyword_val_pairs=[],
                filter_mode="equal",
                include_tags=None,
                exclude_tags=None,
            )]

        if use_all_pairs:
            # 第一次生成并缓存
            if self.cached_all_pairs_train_dicts is None:
                print("Generating all pairs dataset (one-time generation)...")
                full_train_dicts_dict, full_test_dicts = self.dataset_generator.generate_datasets(
                    train_size=train_size,
                    test_size=test_size,
                    filter_conditions=filter_conditions,
                    rhythmicity_pair_conditions=rhythmicity_pair_conditions,
                    use_all_pairs=True,
                )
                self.cached_all_pairs_train_dicts = full_train_dicts_dict
                self.cached_all_pairs_test_dicts = full_test_dicts
                for fc_idx, dicts in full_train_dicts_dict.items():
                    print(f"  Condition {fc_idx}: {len(dicts)} training samples")
                
                self.current_indices = [0 for _ in range(len(filter_conditions))]

            total_conditions = len(filter_conditions)
            total_samples_list = [len(dicts) for dicts in self.cached_all_pairs_train_dicts.values()]
            
            sampled_train_dicts = []

            for i in range(total_conditions):
                start_idx = self.current_indices[i]
                end_idx = start_idx + train_size
            
                if end_idx <= total_samples_list[i]:
                    sampled_train_dicts += self.cached_all_pairs_train_dicts[i][start_idx:end_idx]
                    self.current_indices[i] = end_idx
                else:
                    sampled_train_dicts += self.cached_all_pairs_train_dicts[i][start_idx:] + self.cached_all_pairs_train_dicts[i][:(end_idx - total_samples_list[i])]
                    self.current_indices[i] = end_idx - total_samples_list[i]
            
                # 打印进度信息
                # epochs_to_cover_all = total_samples_list[i] / train_size
                # current_epoch_in_cycle = self.current_index / train_size
                print(
                    f"Condition {i}: ",
                    f"Using samples [{start_idx}:{end_idx % total_samples_list[i]}], "
                    f"Progress: {self.current_indices[i]}/{total_samples_list[i]} "
                    f"({self.current_indices[i]/total_samples_list[i]*100:.1f}%)"
                )

            random.shuffle(sampled_train_dicts)

            train_dataset = DMSDataset(sampled_train_dicts, device=device)
            test_dataset = DMSDataset(self.cached_all_pairs_test_dicts, device=device)
            return train_dataset, test_dataset

        train_dicts, test_dicts = self.dataset_generator.generate_datasets(
            train_size=train_size, test_size=test_size,
            filter_conditions=filter_conditions,
            rhythmicity_pair_conditions=rhythmicity_pair_conditions,
            use_all_pairs=False,
        )

        train_dataset = DMSDataset(train_dicts, device=device)
        test_dataset = DMSDataset(test_dicts, device=device)
        
        return train_dataset, test_dataset
    


# ds = create_datasets(
#     change_nums=[0, 2],
#     permitted_movements=[
#         [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0], 
#     ],
# )
# ds.dataset_generator.channel_sequence_pair_database.statistics(
#     keyword_val_pairs=[("change_num", 0)],
#     filter_mode="equal",
#     include_tags=None,
#     exclude_tags=None,
# )
# ds.dataset_generator.channel_sequence_pair_database.statistics(
#     keyword_val_pairs=[("change_num", 1)],
#     filter_mode="equal",
#     include_tags=None,
#     exclude_tags=None,
# )
# ds.dataset_generator.channel_sequence_pair_database.statistics(
#     keyword_val_pairs=[("change_num", 2)],
#     filter_mode="equal",
#     include_tags=None,
#     exclude_tags=None,
# )

