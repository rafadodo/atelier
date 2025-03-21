# -*- coding: utf-8 -*-
"""
This module contains functions for Operational Modal Analysis (OMA), namely, Peak-Picking,
Frequency Domain Decomposition (FDD), Enhanced FDD (EFDD), and Curve-fitting FDD (CFDD), as
described in the corresponding literature, which can be found in each function docstring.

Functions:
- get_peak_picking_modes(psd, angle_th, mode_idxes): Extracts mode shapes from a PSD matrix.
- get_efdd_segment(sing_vectors, peak_idx, mac_th, sv_num): Identifies frequency indexes for EFDD.
- get_mean_modeshape_efdd(segment_idxes, peak_idx, sing_vectors): Computes a mean mode shape.
- curve_fit_psd_peak(freq, psd, indexes, f_hat): Fits a PSD peak to a 1DOF system model.
"""
import numpy as np

import mac


def get_peak_picking_modes(psd, angle_th, mode_idxes):
    """Extracts mode shapes from a PSD matrix using the peak-picking method (see Reference below).

    Parameters:
        psd (numpy.ndarray, shape (n_freq, n_dof)):
            The power spectral density matrix.
            - n_freq: Number of frequency bins.
            - n_dof: Number of degrees of freedom.
        angle_th (float): The phase difference threshold in degrees.
        mode_idxes (list of int): List of indexes corresponding to mode frequencies.

    Returns:
        numpy.ndarray: A matrix where each column is a mode shape obtained via peak picking.

    Reference:
        A. J. Felber, "Development of a Hybrid Bridge Evaluation System. Ph.D.
        Thesis", University of British Columbia, 1993.
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


def get_efdd_segment(sing_vectors, peak_idx, mac_th, sv_num):
    """Determines the frequency range to be used for estimating the mode properties of a given
    peak in a singular-value curve, as per the Enhanced Frequency Domain Decomposition method (see
    Reference below).

    Parameters:
        sing_vectors (numpy.ndarray, shape (n_freq, n_dof, n_modes)):
            Singular vector matrix obtained from PSD decomposition.
            - n_freq: Number of frequency bins.
            - n_dof: Number of degrees of freedom (DOFs).
            - n_modes: Number of extracted modes.
        peak_idx (int): Index of the mode frequency peak.
        mac_th (float): MAC threshold for defining the frequency segment.
        sv_num (int): Singular vector index corresponding to the mode of interest.

    Returns:
        tuple[int, int]: Lower and upper frequency indices defining the EFDD segment.

    Reference:
        R. Brincker, C. E. Ventura y P. Andersen, "Damping Estimation by Frequency
        Domain Decomposition", Proceedings of the 19th International Modal Analysis
        Conference, 2001.
    """
    sdof_mode = sing_vectors[peak_idx, :, sv_num]
    lower_idx = peak_idx
    mac_value = 1
    while (mac_value>mac_th) & (lower_idx>0):
        lower_idx -= 1
        mac_value = mac.get_mac_value(sdof_mode, sing_vectors[lower_idx, :, sv_num])

    upper_idx = peak_idx
    mac_value = 1
    while (mac_value>mac_th) & (upper_idx<sing_vectors.shape[0]//2):
        upper_idx += 1
        mac_value = mac.get_mac_value(sdof_mode, sing_vectors[upper_idx, :, sv_num])

    return lower_idx, upper_idx


def get_mean_modeshape_efdd(segment_idxes, peak_idx, sing_vectors):
    """Computes the mean mode shape around a system response peak using Enhanced Frequency Domain
    Decomposition (see Reference below).

    Parameters:
        segment_idxes (tuple[int, int]): Indexes defining the frequency segment.
        peak_idx (int): Index of the mode frequency peak.
        sing_vectors (numpy.ndarray, shape (n_freq, n_dof, n_modes)):
            Singular vector matrix obtained from PSD decomposition.
            - n_freq: Number of frequency bins.
            - n_dof: Number of degrees of freedom (DOFs).
            - n_modes: Number of extracted modes.

    Returns:
        numpy.ndarray: The estimated mean mode shape.

    Reference:
        R. Brincker, C. E. Ventura y P. Andersen, "Damping Estimation by Frequency
        Domain Decomposition", Proceedings of the 19th International Modal Analysis
        Conference, 2001.
    """
    peak_modeshape = sing_vectors[peak_idx, :, 0]
    modeshapes = sing_vectors[segment_idxes[0]:segment_idxes[1], :, 0]

    mac_values = np.zeros(len(modeshapes))
    for idx in range(len(modeshapes)):
        mac_values[idx] = mac.get_mac_value(peak_modeshape, modeshapes[idx, :])

    modeshape_efdd = np.sum((modeshapes.T * mac_values).T, axis=0) / np.sum(mac_values)

    return modeshape_efdd


def curve_fit_psd_peak(freq, psd, indexes, f_hat):
    """Fits the prescribed segment of a PSD curve to a squared 1DOF transmisibility, following the
    Curve-fitting Frequency Domain Decomposition method (see Reference below).

    Parameters:
        freq (numpy.ndarray): Frequency array corresponding to the PSD values.
        psd (numpy.ndarray): Power spectral density values.
        indexes (tuple[int, int]): Index range defining the peak segment.
        f_hat (numpy.ndarray): Frequencies for the fitted curve.

    Returns:
        tuple:
            - f_n (float): Estimated natural frequency.
            - xi_n (float): Estimated damping ratio.
            - psd_hat (numpy.ndarray): Fitted PSD curve.

    Reference:
        N.-J. Jacobsen, P. Andersen y R. Brincker, "Applications of Frequency Domain Curve-Fitting
        in the EFDD Technique", Proceedings of the 26th International Modal Analysis Conference,
        2008.
    """
    w = 2*np.pi*freq[indexes[0]:indexes[1]]
    w_hat = 2*np.pi*f_hat
    B = psd[indexes[0]:indexes[1]]
    A = np.vstack((w**4, -B*w**4, -B*w**2)).T
    x = np.linalg.lstsq(A, B, rcond=-1)[0]

    w_n = 1/np.abs(x[1])**0.25
    f_n = w_n/2/np.pi
    c = np.sqrt(np.abs(x[0])*w_n**4)
    xi_n = np.sqrt((x[2]*w_n**2 + 2)/4)

    psd_hat = c**2 * w_hat**4 / ((w_n**2 - w_hat**2)**2 + 4*xi_n**2 * w_n**2 * w_hat**2)

    return f_n, xi_n, psd_hat