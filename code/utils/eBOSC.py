
import numpy as np

from scipy import signal
from scipy.stats import chi2

import pywt


def wavelet(data, fs, f_vec, wavelet_name='morl'):
    dt = 1 / fs
    fc = pywt.central_frequency(wavelet_name)
    scales = fc / (f_vec * dt)
    coefs, freqs = pywt.cwt(data, scales, wavelet_name, sampling_period=dt)
    return coefs, freqs


def fit_bg(power, f_vec, exclude):
    mask = (f_vec < exclude[0]) | (f_vec > exclude[1])
    log_f = np.log10(f_vec[mask])
    log_p = np.log10(power[mask, :].mean(axis=1))
    A, alpha = np.polyfit(log_f, log_p, 1)
    bg_fit = 10 ** (A + alpha * np.log10(f_vec))
    return bg_fit


def detect_rhythms(power, thr_pow, dur_samp, do_dialation=True, kernel=10):
    R = (power >= thr_pow).astype(int)

    threshed = R.copy()

    if do_dialation:
        for i in range(R.shape[0]):
            R[i] = np.convolve(R[i], np.ones(kernel), mode='same') >= 1
    
    dialated = R.copy()

    max_lengths = []

    for i, d in enumerate(dur_samp):
        start = 0
        max_length = 0
        while start < R.shape[1]:
            if R[i, start] == 1:
                end = start
                while end < R.shape[1] and R[i, end] == 1:
                    end += 1
                length = end - start
                if length < d:
                    R[i, start:end] = 0
                    length = 0
                elif do_dialation:
                    R[i, start:start+kernel-1] = 0
                    R[i, end-kernel+1:end] = 0
                    length = end - start - 2 * kernel + 2
                else:
                    length = end - start
                start = end
                max_length = max(max_length, length)
            else:
                start += 1
        max_lengths.append(max_length)

    return threshed, dialated, R, max_lengths


def eBOSC(data, eBOSC_params, inter_onset_interval):

    fs = eBOSC_params["fs"]
    dur_thr = eBOSC_params["dur_thr"]
    alpha = eBOSC_params["alpha"]
    conf = eBOSC_params["conf"]
    f_vec = eBOSC_params["f_vec"]

    coefs, freqs = wavelet(data, fs, f_vec)

    power = coefs ** 2
    ampls = np.abs(coefs)  # amplitude

    alpha_center = 1000.0 / inter_onset_interval
    alpha_rng = (alpha_center - alpha, alpha_center + alpha)

    bg = fit_bg(power, f_vec, exclude=alpha_rng)

    df = 2
    thr_pow = bg[:, None] * chi2.ppf(conf, df)

    kernel = 10
    dur_samp = np.round(dur_thr * fs / f_vec).astype(int)

    do_dialation = True

    threshed, dialated, R, max_lengths = detect_rhythms(power, thr_pow, dur_samp, do_dialation=do_dialation, kernel=kernel)

    return freqs, ampls, threshed, dialated, R, max_lengths


