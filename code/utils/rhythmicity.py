
import numpy as np

from utils.eBOSC import eBOSC
from typing import Dict


# rhythmicity_bin = [1., 0.875, 0.75, 0.625, 0.5]
rhythmicity_bin = [1., 0.93, 0.86, 0.79, 0.72, 0.65, 0.58, 0.51]

def compute_fstim_cv(onsets_abs: np.ndarray, fs: float = 100) -> Dict[str, np.ndarray]:
    """
        onsets_abs: (n_trial, 6) absolute onset indices in steps
        fs: sampling rate (Hz), e.g. 100 for dt=10ms

        Returns:
        iois_steps: (n_trial, 5) onset-to-onset intervals in steps
        iois_sec:   (n_trial, 5) in seconds
        T_steps:    (n_trial,) mean IOI in steps
        T_sec:      (n_trial,) mean IOI in seconds
        f_stim:     (n_trial,) Hz = 1/T_sec = fs/T_steps
        CV_IOI:     (n_trial,) std(IOI)/mean(IOI)  (ddof=0)
        R:          (n_trial,) 1 - CV_IOI
    """
    onsets_abs = np.asarray(onsets_abs).astype(int)
    iois_steps = np.diff(onsets_abs, axis=1)

    T_steps = iois_steps.mean(axis=1)
    iois_sec = iois_steps / fs
    T_sec = T_steps / fs

    f_stim = 1.0 / (T_sec + 1e-12)

    cv = iois_steps.std(axis=1) / (T_steps + 1e-12)
    R = 1.0 - cv

    return dict(iois_steps=iois_steps, iois_sec=iois_sec, T_steps=T_steps, T_sec=T_sec, f_stim=f_stim, CV_IOI=cv, R=R)



def identify_rhythmic_neurons(
        data, hidden_size, illegal_indices,  
        phase, stimulus_duration, inter_onset_interval, time_step, eBOSC_params,
        threshed_val_thres=150, freqs_band=(2, 4),
    ):

    predictive_indices = []
    rhythmic_indices = []

    threshed_vals = []
    neuron_max_lengths = []

    n_neuron = hidden_size
    for neuron_idx in range(n_neuron):
        for rhythm_plt_idx, rhythm_data_idx in enumerate([0]):
            if neuron_idx in illegal_indices:
                continue

            n_trial = data[rhythm_data_idx].shape[0]

            use_data = data[rhythm_data_idx][:, neuron_idx, :].mean(axis=0)            # average over trials, single neuron
            sem = data[rhythm_data_idx][:, neuron_idx, :].std(axis=0) / np.sqrt(n_trial)

            average_activity = use_data.mean()

            t = np.arange(use_data.shape[0])  # time steps

            freqs, ampls, threshed, dialated, R, max_lengths = eBOSC(use_data, eBOSC_params, inter_onset_interval)
            # ampls: (n_frequencies, n_time) -> (72, 530)

            sieve = "rhythmic spectrum"

            if sieve == "stim 7 occurrence":

                pass

                # relative_thres = 0.6
                # absolute_thres = 0.05
                # extended_steps_on = 0
                # extended_steps_off = 5

                # band_indices = np.where((freqs >= freqs_band[0]) & (freqs <= freqs_band[1]))[0]

                # stim_6_onset = phase['Delay'][0] - stimulus_duration // time_step
                # stim_6_offset = phase['Delay'][0]

                # stim_7_onset = stim_6_onset + inter_onset_interval // time_step
                # stim_7_offset = stim_6_offset + inter_onset_interval // time_step


                # ampl_stim_6 = ampls[band_indices, (stim_6_onset-extended_steps_on):(stim_6_offset+extended_steps_off)]
                # acti_stim_7 = use_data[(stim_7_onset-extended_steps_on):(stim_7_offset+extended_steps_off)]
                # ampl_stim_7 = ampls[band_indices, (stim_7_onset-extended_steps_on):(stim_7_offset+extended_steps_off)]

                # def method(arr):
                #     return np.max(arr)

                # if (method(ampl_stim_7) > method(ampl_stim_6) * relative_thres) and (method(ampl_stim_7) > absolute_thres) and \
                # (np.any(acti_stim_7 > average_activity / 2)): 
                #     predictive_indices.append(neuron_idx)

            if sieve == "rhythmic spectrum":

                band_indices = np.where((freqs >= freqs_band[0]) & (freqs <= freqs_band[1]))[0]

                threshed_val = threshed[band_indices, phase['Sample'][0]:phase['Delay'][1]].sum(axis=1).max()
                threshed_vals.append(int(threshed_val))
                if (threshed_val > threshed_val_thres): 
                    rhythmic_indices.append(neuron_idx)
                
                # max_lengths_band = [max_lengths[i] for i in band_indices]
                # max_length = max(max_lengths_band)
                # neuron_max_lengths.append(max_length)
                # if max_length >= threshed_val_thres:
                #     rhythmic_indices.append(neuron_idx)


    print(sorted(threshed_vals))
    # print(sorted(neuron_max_lengths))
    
    return predictive_indices, rhythmic_indices







def wrap_angle(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x + np.pi) % (2 * np.pi) - np.pi


def build_theta_from_onsets(onsets_abs: np.ndarray, n_time: int):
    """
    theta(t) unwrapped:
      - between consecutive onsets: linear +2π
      - before first onset: extrapolate using first IOI
      - after last onset: extrapolate using last IOI
    Returns:
      theta_u: (n_trial, n_time)
      theta_w: (n_trial, n_time) wrapped
    """
    onsets_abs = np.asarray(onsets_abs).astype(int)
    n_trial = onsets_abs.shape[0]
    theta = np.zeros((n_trial, n_time), dtype=float)

    for tr in range(n_trial):
        ons = onsets_abs[tr]
        iois = np.diff(ons)
        if iois.size == 0:
            continue
        # fix degenerate
        if np.any(iois <= 0):
            pos = iois[iois > 0]
            fill = int(np.median(pos)) if pos.size else 40
            iois = np.where(iois > 0, iois, fill)

        phase_at_onset = 2 * np.pi * np.arange(ons.size)

        # fill between onsets
        for k in range(ons.size - 1):
            t0, t1 = int(ons[k]), int(ons[k+1])
            if t1 <= t0:
                continue
            phi0, phi1 = phase_at_onset[k], phase_at_onset[k+1]
            theta[tr, t0:t1] = np.linspace(phi0, phi1, t1 - t0, endpoint=False)

        # before first onset
        t_first = int(ons[0])
        T0 = int(iois[0])
        if t_first > 0:
            theta[tr, :t_first] = np.linspace(-2*np.pi*(t_first/(T0+1e-12)), 0.0, t_first, endpoint=False)

        # after last onset
        t_last = int(ons[-1])
        Tn = int(iois[-1])
        if t_last < n_time:
            theta[tr, t_last:] = phase_at_onset[-1] + 2*np.pi*np.arange(n_time - t_last) / (Tn + 1e-12)

    theta_w = wrap_angle(theta)
    return theta, theta_w



from scipy.signal import butter, filtfilt, hilbert


def bandpass_1d(x: np.ndarray, fs: float, f_lo: float, f_hi: float, order: int = 4):
    nyq = 0.5 * fs
    f_lo = max(0.5, float(f_lo))
    f_hi = min(nyq - 0.5, float(f_hi))
    if f_hi <= f_lo:
        return None
    b, a = butter(order, [f_lo/nyq, f_hi/nyq], btype="band")
    # x length should be long enough; for global sample window we ensure tail
    return filtfilt(b, a, x, method="pad")


def compute_rhythmicity_metrics(neural_data, sample_onset, fs, target_freq, time_window=None, baseline_window=None, mode="hilbert"):
    """
    compute the intrinsic rhythmicity metrics. including Induced Power, Evoked Power and PLV.
    
    Parameters:
    - neural_data: (n_trials, n_timepoints)
    - sample_onset: (n_trials, n_stim)
    - fs: sampling rate
    - target_freq: target frequency
    - time_window: (start_idx, end_idx)
    - baseline_window: (start_idx, end_idx)
    - mode: "morlet" or "hilbert"
    """
    n_trials, n_timepoints = neural_data.shape
    if time_window is None:
        start, end = int(n_timepoints*0.1), int(n_timepoints*0.9)
    else:
        start, end = time_window
    
    if mode == "hilbert":
        ## Hilbert transform
        f_lo = target_freq - 1.0
        f_hi = target_freq + 1.0
        analytic_signal_all = np.zeros((n_trials, n_timepoints), dtype=complex)
        
        for trial_i in range(n_trials):
            filtered = bandpass_1d(neural_data[trial_i, :], fs, f_lo, f_hi)
            if filtered is None:
                continue
            analytic_signal_all[trial_i, :] = hilbert(filtered)

        erp_signal = np.mean(neural_data, axis=0)
        erp_filtered = bandpass_1d(erp_signal, fs, f_lo, f_hi)
        erp_analytic = hilbert(erp_filtered) if erp_filtered is not None else np.zeros_like(erp_signal, dtype=complex)
        
        ## Induced Power (abs)
        power_per_trial = np.abs(analytic_signal_all) ** 2
        induced_power_series = np.mean(power_per_trial, axis=0)

        ## PLV
        theta_u, theta_w = build_theta_from_onsets(sample_onset, n_time=n_timepoints)
        theta_seg = theta_u[:, start:end]                                   # external phase
        phase_all_trials = np.angle(analytic_signal_all)[:, start:end]      # internal phase
        diff_exp = np.exp(1j * (phase_all_trials - theta_seg))
        plv_series = np.abs(np.nanmean(diff_exp, axis=0))
        
        ## Evoked Power (abs)
        evoked_power_series = np.abs(erp_analytic) ** 2


    elif mode == "morlet":
        ## Morlet wavelet
        n_cycles = 6 
        sigma = n_cycles / (2 * np.pi * target_freq)
        t_wavelet = np.arange(-2.0, 2.0, 1/fs) 
    
        if len(t_wavelet) % 2 == 0:
            t_wavelet = t_wavelet[:-1]

        sine_wave = np.exp(1j * 2 * np.pi * target_freq * t_wavelet)
        gaussian_win = np.exp(-t_wavelet**2 / (2 * sigma**2))
        wavelet = sine_wave * gaussian_win
        
        ## FFT convolution
        n_conv = n_timepoints + len(t_wavelet) - 1
        start_ind = (len(t_wavelet) - 1) // 2
        end_ind = start_ind + n_timepoints
        
        wavelet_fft = np.fft.fft(wavelet, n_conv)
        analytic_signal_all = np.zeros((n_trials, n_timepoints), dtype=complex)
        
        for trial_i in range(n_trials):
            data_fft = np.fft.fft(neural_data[trial_i, :], n_conv)
            conv_res = np.fft.ifft(wavelet_fft * data_fft)
            analytic_signal_all[trial_i, :] = conv_res[start_ind:end_ind]

        erp_signal = np.mean(neural_data, axis=0)
        
        erp_fft = np.fft.fft(erp_signal, n_conv)
        erp_conv_res = np.fft.ifft(wavelet_fft * erp_fft)
        erp_analytic = erp_conv_res[start_ind:end_ind]
        
        ## Induced Power (abs)
        power_per_trial = np.abs(analytic_signal_all) ** 2
        induced_power_series = np.mean(power_per_trial, axis=0)

        ## PLV
        theta_u, theta_w = build_theta_from_onsets(sample_onset, n_time=n_timepoints)
        theta_seg = theta_u[:, start:end]                                   # external phase
        phase_all_trials = np.angle(analytic_signal_all)[:, start:end]      # internal phase
        diff_exp = np.exp(1j * (phase_all_trials - theta_seg))
        plv_series = np.abs(np.nanmean(diff_exp, axis=0))
        
        ## Evoked Power (abs)
        evoked_power_series = np.abs(erp_analytic) ** 2
    
    ## summary
    if baseline_window is not None:
        baseline_start, baseline_end = baseline_window
        baseline_induced_power = np.mean(induced_power_series[baseline_start:baseline_end])
        baseline_evoked_power = np.mean(evoked_power_series[baseline_start:baseline_end])
        
        induced_power_series = induced_power_series / (baseline_induced_power + 1e-12)
        evoked_power_series = evoked_power_series / (baseline_evoked_power + 1e-12)
        
    scalar_induced_power = np.mean(induced_power_series[start:end])
    scalar_evoked_power = np.mean(evoked_power_series[start:end])
    scalar_plv = np.mean(plv_series[start:end])
    
    return {
        "induced_power_score": scalar_induced_power,
        "evoked_power_score": scalar_evoked_power,
        "plv_score": scalar_plv
    }, {
        "induced_power_series": induced_power_series,
        "evoked_power_series": evoked_power_series,
        "plv_series": plv_series,
        "phase_data": phase_all_trials
    }








