# -*- coding: utf-8 -*-
"""This module contains the functions necessary to perform the RFP modal
analysis method as described in Richardson & Formenti [1] [2] [3].

orthogonal_polynomials: computes the Forsythe polynomials for a given FRF

rfp: computes the modal constants for a given FRF.

grfp_denominator: computes the common denominator polynomial for a given
    set of FRFs.

grfp_parameters: computes the modal parameters for a set of FRFs.

[1] Richardson, M. H. & Formenti D. L. "Parameter estimation from frequency
response measurements using rational fraction polynomials", 1st IMAC Conference,
Orlando, FL, 1986.

[2] Richardson, M. H. "Global frequency and damping estimates from frequency
response measurements", 4th IMAC Conference, Los Angeles, CA, 1986.

[3] Richardson, M. H. & Formenti D. L. "Parameter estimation from frequency
response measurements using rational fraction polynomials", 1st IMAC Conference,
Orlando, FL, 1986.
"""

import numpy as np
from scipy import signal


def orthogonal_polynomials(frf, omega, weights, order):
    """Compute the Forsythe orthogonal polynomials that approximate the
    frequency response function FRF over the frequency range omega, to be used
    in the RFP method as described by Richardson and Formenti [1].
    
    Arguments:
        FRF (array): Frequency Response Function vector ()
        omega (array): Angular velocity range (rad/s)
        weights (str): "ones" for the Phi matrix, 
                        "frf" for the theta matrix.
        order (int): Polynomial degree

    Returns:
        P (array): Matrix of the polynomials evaluated at the given 
                    frequencies.
        Conv (array): Matrix that converts Forsythe coefficients to standard
                        polynomial coefficients.

    [1] Richardson, M. H. & Formenti D. L. "Parameter estimation from frequency
    response measurements using rational fraction polynomials", 1st IMAC Conference,
    Orlando, FL, 1986.
    """
    
    P = np.zeros((len(omega), order+1), dtype=complex)
    Conv = np.zeros((order+1, order+1))
    R = np.zeros((len(omega), order+2))
    Conv_ext = np.zeros((order+1, order+2))
    
    if weights=="ones":
        q = np.ones(len(omega)) # weighting values for phi matrix
    elif weights=="frf":
        q = np.abs(frf)**2    # weighting vuales for theta matrix
    else:
        raise Exception('Invalid weights.')

    R[:, 0] = np.zeros(len(omega))
    R[:, 1] = 1 / np.sqrt(2 * np.sum(q))

    Conv_ext[0,1] = 1 / np.sqrt(2 * np.sum(q))

    for k in range(2, order+2):
        V_km1 = 2*np.sum(omega * R[:, k-1] * R[:, k-2] * q)
        S_k = omega * R[:, k-1] - V_km1 * R[:, k-2]
        D_k = np.sqrt(2 * np.sum(S_k**2 * q))
        
        R[:, k] = S_k / D_k
        Conv_ext[:, k] = (1/D_k) *(-V_km1 * Conv_ext[:, k-2] + \
                          np.concatenate([[0], Conv_ext[:-1, k-1]]))

    j_k =  np.zeros((order+1,1), dtype=complex)
    for k in range(order+1):
        P[:, k] = 1j**k * R[:, k+1]
        j_k[k] = 1j**k

    Conv = Conv_ext[:, 1:]
    Conv = (j_k @ j_k.conj().T).T * Conv
        
    return P, Conv

def rfp(frf, omega, n_dof):
    """Computes estimates for the modal parameters of the given FRF, in the
    frequency range given by omega, modeling it as an n_dof degrees of freedom
    system, and following the RFP method as described by Richardson and Formenti [1].

    Arguments:
        frf (numpy complex array):
            - Frequency Response Function vector (receptance).
        omega (numpy array):
            - Angular velocity range (rad/s)
        n_dof (int):
            - Number of degrees of freedom (modes) to consider for the
            estimation

    Returns:
        alpha (numpy complex array):
            - Estimated receptance FRF in the given frequency range.
        modal_params (array list):
            - Modal parameter  for the estimated FRF Modal parameter list:
                [freq_n, xi_n, modal_mag_n, modal_ang_n]

    [1] Richardson, M. H. & Formenti D. L. "Parameter estimation from frequency
    response measurements using rational fraction polynomials", 1st IMAC Conference,
    Orlando, FL, 1986.
    """

    omega_norm = omega / np.max(omega) # omega normalization
    m = 2*n_dof - 1 # number of polynomial terms in numerator
    n = 2*n_dof # number of polynomial terms in denominator
    d = np.zeros(n+1) # Orthogonal denominator polynomial coefficients

    # computation of Forsythe orthogonal polynomials
    Phi, Conv_A = orthogonal_polynomials(frf, omega_norm, 'ones', m)
    Theta, Conv_B = orthogonal_polynomials(frf, omega_norm, 'frf', n)

    T = np.diag(frf) @ Theta[:, :-1]
    W = frf * Theta[:, -1]
    X = -2 * np.real(Phi.T.conj() @ T)
    H = 2 * np.real(Phi.T.conj() @ W)

    d[:-1] = -np.linalg.inv(np.eye(X.shape[1]) - X.T @ X) @ X.T @ H
    d[-1] = 1
    c = H - X @ d[:-1] # Orthogonal numerator polynomial coefficients

    # calculation of the estimated FRF (alpha)
    numer = Phi @ c
    denom = Theta @ d
    alpha = numer / denom

    a = np.flipud(Conv_A @ c) # Standard polynomial numerator coefficients
    b = np.flipud(Conv_B @ d) # Standard polynomial denominator coefficients

    # Calculation of the poles and residues
    res, pol, _ = signal.residue(a, b)
    residues = res[::2] * np.max(omega)
    poles = pol[::2] * np.max(omega)

    freq_n = np.abs(poles)/2/np.pi # Natural frequencies (rad/sec)
    xi_n = -np.real(poles) / np.abs(poles) # Damping ratios
    modal_const = 1j*2*residues*np.imag(poles)
    modal_mag_n = np.abs(modal_const) # Modal constant magnitude
    modal_ang_n = np.angle(modal_const) # Modal constant phase

    modal_params = [freq_n, xi_n, modal_mag_n, modal_ang_n] # Modal Parameters

    return modal_params, alpha

def grfp_denominator(frf, omega, n_modes):
    """Computes an estimate of the denominator polynomial shared by all the FRFs
    given by the columns of "frf", which correspond to the frequency range given by
    omega, assuming that the number of modes contributing is "n_modes", and following
    the GRFP method as described by Richardson [1].

    Arguments:
        - frf (numpy complex array):
            - Frequency Response Function matrix. Each column contains the FRF
            for a particular DOF, with all of them corresponding the reference or
            input DOF.
        - omega (numpy array):
            - Angular velocity range (rad/s).
        - n_modes (int):
            - Number of modes to assume for the estimation.

    Returns:
        - denominator (numpy complex array):
             - Estimated denominator polynomial.
        - denominator_coeff (numpy complex array):
            - Estimated denominator polynomial coefficients.

    [1] Richardson, M. H. "Global frequency and damping estimates from frequency
    response measurements", 4th IMAC Conference, Los Angeles, CA, 1986.
    """

    n_dof = frf.shape[1] # number of frf measurements (degrees of freedom)
    w_norm = omega/np.max(omega) # normalized angular frequency range
    w_j = 1j*w_norm # complex normalized angular frequency range
    m = 2*n_modes - 1 # number of polynomial terms in numerator
    n = 2*n_modes # number of polynomial terms in denominator

    U = np.zeros((n_dof, n, n), dtype=complex)
    V = np.zeros((n_dof, n), dtype=complex)
    for dof in range(n_dof):
        Phi, Conv_A = orthogonal_polynomials(frf[:, dof], w_norm, 'ones', m)
        Theta, Conv_B = orthogonal_polynomials(frf[:, dof], w_norm, 'frf', n)
        T = np.diag(frf[:, dof]) @ Theta[:, :-1]
        W = frf[:, dof] * Theta[:, -1]
        X = -2 * np.real(Phi.T.conj() @ T)
        H = 2 * np.real(Phi.T.conj() @ W)
        U[dof, :, :] = (np.eye(X.shape[1]) - X.T @ X) @ \
                        np.linalg.inv(Conv_B)[:-1, :-1]
        V[dof, :] = X.T @ H

    A = np.sum(U@U, axis=0)
    b = np.sum(U@V[:, :, np.newaxis], axis=0)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    denominator_coeff = np.flipud(x[:, 0])
    denominator = np.polyval(denominator_coeff, w_j)
    return denominator, denominator_coeff

def grfp_parameters(frf, omega, denom, denom_coeff, n_modes):
    """Computes an estimate of the modal parameters for each of the FRFs given by
    the columns of "frf", which correspond to the frequency range given by omega,
    following the GRFP method described by Richardson and Formenti [1]. It is assumed
    that "n_modes" is the number of modes contributing, and that the common denominator
    polynomial shared by all the FRFs is given by "denom", formed by the coefficients
    "denom_coeff".

    Arguments:
        - frf (numpy complex array):
            - Frequency Response Function matrix. Each column contains the FRF
            for a particular DOF, with all of them corresponding the reference or
            input DOF.
        - omega (numpy array):
            - Angular velocity range (rad/s).
        - n_modes (int):
            - Number of modes to assume for the estimation.

    Returns:
        alpha (numpy complex array):
            - Estimated receptance FRF in the given frequency range.
        modal_params (array list):
            - Modal parameter  for the estimated FRF Modal parameter list:
                [freq_n, xi_n, modal_mag_n, modal_ang_n]

    [1] Richardson, M. H. & Formenti, D. L. "Global curve fitting of frequency response
    measurements using the Rational Fraction Polynomial method", 3rd IMAC Conference,
    Orlando, FL, 1985.
    """

    m = 2*n_modes - 1 # number of polynomial terms in numerator
    n_dof = frf.shape[1] # number of frf measurements (degrees of freedom)
    w_norm = omega/np.max(omega) # normalized angular frequency range
    w_j = 1j*w_norm # complex normalized angular frequency range
    c = np.zeros((m+1, n_dof)) # orthogonal numerator polynomial coefficients
    # standard numerator polynomial coefficients
    numer_coef = np.zeros((m+1, n_dof), dtype=complex)
    numer = np.zeros((len(w_norm), n_dof), dtype=complex) # numerator polynomials
    alpha = np.zeros((len(w_norm), n_dof), dtype=complex) # FRF estimations
    residues_norm = np.zeros((m, n_dof), dtype=complex) # frecuency-normalized residues
    poles_norm = np.zeros((m, n_dof), dtype=complex) # frequency-normalized poles

    Z, Conv_A = orthogonal_polynomials(1/denom, w_norm, 'frf', m)
    X = np.diag(1/denom)@Z
    for dof in range(n_dof):
        c[:, dof] = 2*np.real(X.conj().T@frf[:, dof])
        numer_coef[:, dof] = np.flipud(Conv_A@c[:, dof])
        numer[:, dof] = np.polyval(numer_coef[:, dof], w_j)
        alpha[:, dof] = numer[:, dof] / denom
        residues_norm[:, dof], poles_norm[:, dof], _ = signal.residue(numer_coef[:, dof],
                                                                      denom_coeff)

    xi_n_raw = -np.real(poles_norm*np.max(omega))/np.abs(poles_norm*np.max(omega))
    # modes with unitary damping coefficient are discarded
    physical_modes_idx = (abs(xi_n_raw[:, 0])!=1)
    residues = residues_norm[physical_modes_idx, :][::2]*np.max(omega)
    poles = poles_norm[physical_modes_idx, :][::2]*np.max(omega)

    freq_n = np.abs(poles[:, 0])/2/np.pi
    xi_n = -np.real(poles[:, 0])/np.abs(poles[:, 0])
    modal_const = 1j*2*residues*np.imag(poles)
    modal_mag_n = np.abs(modal_const) # Modal constant magnitude
    modal_ang_n = np.angle(modal_const) # Modal constant phase

    modal_params = [freq_n, xi_n, modal_mag_n, modal_ang_n]
    return alpha, modal_params