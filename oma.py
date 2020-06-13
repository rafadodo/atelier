import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import mac

def get_efdd_segment(s_vec, peak_idx, mac_th):
    """Return the frequency indexes around peak_idx where the psd
    matrix having s_vec as its singular vector matrix can be
    considered as the psd of the 1DOF system, corresponding to the
    mode shape that the first singular vector takes at peak_idx.

    A MAC above the threshold mac_th is used as a criterion."""

    sdof_mode = s_vec[peak_idx, :, 0]
    lower_idx = peak_idx
    mac_value = 1
    while (mac_value>mac_th) & (lower_idx>0):
        lower_idx -= 1
        mac_value = mac.get_MAC(sdof_mode, s_vec[lower_idx, :, 0])

    upper_idx = peak_idx
    mac_value = 1
    while (mac_value>mac_th) & (upper_idx<s_vec.shape[0]//2):
        upper_idx += 1
        mac_value = mac.get_MAC(sdof_mode, s_vec[upper_idx, :, 0])

    return lower_idx, upper_idx

def get_damp_from_decay(decay):
    """Calculate the damping of a decaying signal
    from its logarithmic decrement, obtained by
    least squares."""

    peak_ind = np.array([m for m in signal.argrelmax(abs(decay), order=1)]).flatten()
    log_peak_ratios = np.zeros(len(peak_ind)-1)
    for i in range(len(peak_ind)-1):
        log_peak_ratios[i] = 2*np.log(abs(decay[peak_ind[0]] / decay[peak_ind[i+1]]))
    peak_nums = np.linspace(1, len(log_peak_ratios), len(log_peak_ratios))

    A = np.vstack([peak_nums, np.ones(len(peak_nums))]).T
    b = log_peak_ratios
    plt.scatter(A[:,0], b)

    m, c = np.linalg.lstsq(A, b, rcond=None)[0]
    resid = np.linalg.lstsq(A, b, rcond=None)[1][0]
    R2 = 1 - resid / (b.size * b.var())
    plt.plot(peak_nums, c + m*peak_nums)

    damp = m/np.sqrt(m**2 + 4*np.pi**2)
    return damp, R2

def get_freq_from_signal(timestamps, values):
    """Calculate the frequency of a signal by
    averaging the times between zero crossings."""

    zero_cross_idx = np.where(np.diff(np.sign(values.real)))[0]
    crossing_nums = np.arange(len(zero_cross_idx))+1
    time_intervals_doubled = 2*timestamps[zero_cross_idx]

    A = np.vstack([time_intervals_doubled, np.ones(len(time_intervals_doubled))]).T
    b = crossing_nums

    m, c = np.linalg.lstsq(A, b, rcond=None)[0]
    resid = np.linalg.lstsq(A, b, rcond=None)[1][0]
    R2 = 1 - resid / (b.size * b.var())

    freq = m
    return freq

def curve_fit_psd_peak(f, psd, indexes, f_hat):
    """Fits the segment of the psd curve between the given indexes to a 1DOF PSD,
    considering it as the square of a 1DOF transmisibility, |H(f)|^2.
    The frequencies of the given psd are contained in f, while the frequencies for
    the desired approximation are given in f_hat.
    
    Returns the mode frequency f_n, the mode damping xi_n, and the fitted curve psd_hat.
    """
    w = 2*np.pi*f[indexes[0]:indexes[1]]
    w_hat = 2*np.pi*f_hat
    B = psd[indexes[0]:indexes[1]]
    A = np.vstack((w**4, -B*w**4, -B*w**2)).T    
    x = np.linalg.lstsq(A, B, rcond=None)[0]
    
    w_n = 1/x[1]**0.25
    f_n = w_n/2/np.pi
    c = np.sqrt(x[0]*w_n**4)
    xi_n = np.sqrt(abs(x[2]*w_n**2 + 2)/4)
    psd_hat = c**2 * w_hat**4 / \
              ((w_n**2 - w_hat**2)**2 + 4*xi_n**2 * w_n**2 * w_hat**2)
    return f_n, xi_n, psd_hat