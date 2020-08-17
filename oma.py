import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import mac

def get_peak_picking_modes(psd, angle_th, mode_idxes):
    """Obtains the mode shapes from the given PSD matrix and the mode indexes,
    using the given angle threshold for the phase difference.
    The rows and columns of the PSD matrix shall correspond to the frequencies
    and DOFs respectively.
    """
    total_dof = psd.shape[1]
    total_modes = len(mode_idxes)
    modes_pp = np.zeros((total_dof, total_modes))

    for mode in range(total_modes):
        idx = mode_idxes[mode]
        for dof in range(total_dof):
            cross_power = psd[idx, dof] * np.conj(psd[idx, 0])
            ang = abs(np.angle(cross_power, deg=True))
            sign = 0
            if 0 <= ang <= angle_th:
                sign = 1
            elif (180 - angle_th) <= ang <= 180:
                sign = -1
            modes_pp[dof, mode] = sign * abs(psd[idx, dof]) / abs(psd[idx, 0])

    # Normalization
    for col in range(modes_pp.shape[1]):
        modes_pp[:,col] = modes_pp[:,col]/max(abs(modes_pp[:,col]))

    return modes_pp

def get_efdd_segment(s_vec, peak_idx, mac_th):
    """Returns the frequency indexes around peak_idx where the psd
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
    """Calculates the damping of a decaying signal
    from its logarithmic decrement, obtained by
    least squares."""

    peak_ind = np.array([m for m in signal.argrelmax(abs(decay), order=1)]).flatten()
    log_peak_ratios = np.zeros(len(peak_ind)-1)
    for i in range(len(peak_ind)-1):
        log_peak_ratios[i] = 2*np.log(abs(decay[peak_ind[0]] / decay[peak_ind[i+1]]))
    peak_nums = np.linspace(1, len(log_peak_ratios), len(log_peak_ratios))

    A = np.vstack([peak_nums, np.ones(len(peak_nums))]).T
    b = log_peak_ratios

    m, c = np.linalg.lstsq(A, b, rcond=None)[0]
    resid = np.linalg.lstsq(A, b, rcond=None)[1][0]
    R2 = 1 - resid / (b.size * b.var())
    damp = m/np.sqrt(m**2 + 4*np.pi**2)
    return damp, R2, A, b, c, m

def get_freq_from_signal(timestamps, values):
    """Calculates the frequency of a signal by
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

def get_mean_modeshape_efdd(segment_idxes, peak_idx, s_vectors):
    peak_modeshape = s_vectors[peak_idx, :, 0]
    modeshapes = s_vectors[segment_idxes[0]:segment_idxes[1], :, 0]

    mac_values = np.zeros(len(modeshapes))
    for idx in range(len(modeshapes)):
        mac_values[idx] = mac.get_MAC(peak_modeshape, modeshapes[idx, :])

    modeshape_efdd = np.sum((modeshapes.T * mac_values).T, axis=0) / np.sum(mac_values)
    return modeshape_efdd

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